import copy
from components.episode_buffer import EpisodeBatch
from functools import partial
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.flex_qmix import FlexQMixer, LinearFlexQMixer
from components.action_selectors import parse_avail_actions
import torch as th
import torch.nn.functional as F
import torch.distributions as D
from torch.optim import RMSprop, Adam


class QLearner:
    def __init__(self, mac, scheme, logger, args, elite_buffer=None):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.use_copa = self.args.hier_agent['copa']
        self.elite_buffer = elite_buffer  # 精英缓存池（用于蒸馏损失）

        self.params = list(mac.parameters())
        if self.use_copa:
            self.params += list(self.mac.coach.parameters())
            if self.args.hier_agent['copa_vi_loss']:
                self.params += list(self.mac.copa_recog.parameters())

        self.last_target_update_episode = 0
        self.last_alloc_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            elif args.mixer == "flex_qmix":
                assert args.entity_scheme, "FlexQMixer only available with entity scheme"
                self.mixer = FlexQMixer(args)
            elif args.mixer == "lin_flex_qmix":
                assert args.entity_scheme, "FlexQMixer only available with entity scheme"
                self.mixer = LinearFlexQMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps,
                                 weight_decay=args.weight_decay)

        if self.args.hier_agent["task_allocation"] in ["aql", "a2c", "ppo"]:
            self.alloc_pi_params = list(mac.alloc_pi_params())
            if self.args.hier_agent["alloc_opt"] == "rmsprop":
                OptClass = partial(RMSprop, alpha=args.optim_alpha)
            elif self.args.hier_agent["alloc_opt"] == "adam":
                OptClass = Adam
            else:
                raise Exception("Optimizer not recognized")
            self.alloc_pi_optimiser = OptClass(
                params=self.alloc_pi_params, lr=args.lr, eps=args.optim_eps,
                weight_decay=args.weight_decay)
            self.alloc_q_params = list(mac.alloc_q_params())
            self.alloc_q_optimiser = OptClass(
                params=self.alloc_q_params, lr=args.lr, eps=args.optim_eps,
                weight_decay=args.alloc_q_weight_decay)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.log_alloc_stats_t = -self.args.learner_log_interval - 1

    def _get_mixer_ins(self, batch):
        if not self.args.entity_scheme:
            return (batch["state"][:, :-1],
                    batch["state"][:, 1:])
        else:
            entities = []
            bs, max_t, ne, ed = batch["entities"].shape
            entities.append(batch["entities"])
            if self.args.entity_last_action:
                last_actions = th.zeros(bs, max_t, ne, self.args.n_actions,
                                        device=batch.device,
                                        dtype=batch["entities"].dtype)
                last_actions[:, 1:, :self.args.n_agents] = batch["actions_onehot"][:, :-1]
                entities.append(last_actions)

            entities = th.cat(entities, dim=3)
            mix_ins = {"entities": entities[:, :-1],
                       "entity_mask": batch["entity_mask"][:, :-1]}
            targ_mix_ins = {"entities": entities[:, 1:],
                            "entity_mask": batch["entity_mask"][:, 1:]}
            if self.args.multi_task:
                # use same subtask assignments for prediction and target
                mix_ins["entity2task_mask"] = batch["entity2task_mask"][:, :-1]
                targ_mix_ins["entity2task_mask"] = batch["entity2task_mask"][:, :-1]
            return mix_ins, targ_mix_ins

    def _make_meta_batch(self, batch: EpisodeBatch):
        # 关键修复：在函数开头统一所有张量形状为(bs, ts, 1)
        # 避免后续shape error和广播问题
        # 这是提高代码可靠性的关键步骤
        
        reward = batch['reward']
        terminated = batch['terminated'].float()
        reset = batch['reset'].float()
        mask = batch['filled'].float()
        
        # 统一reward形状为(bs, ts, 1) - 强制全局标量
        # 关键修复4：对A2C上层，最稳妥的做法是只接受(bs, ts, 1)的global reward
        # 如果不是，发出警告但继续处理（避免silently学歪）
        reward_original_shape = reward.shape
        if reward.dim() == 2:
            reward = reward.unsqueeze(-1)  # (bs, ts) -> (bs, ts, 1)
        elif reward.dim() == 3:
            if reward.shape[2] == 1:
                # 已经是(bs, ts, 1)，保持
                pass
            else:
                # 关键修复4：如果reward不是(bs, ts, 1)，发出警告
                # 这可能是语义错误（比如应该是per-task reward但被sum掉了）
                import warnings
                warnings.warn(
                    f"Reward shape {reward_original_shape} is not (bs, ts, 1). "
                    f"Will sum over last dimension to get global reward. "
                    f"This may not match your intended high-level reward semantics!",
                    UserWarning
                )
                # 如果是(bs, ts, n_agents)等，强制聚合为全局标量
                # 关键修复：使用sum而不是mean，因为MARL环境通常是团队奖励
                # 如果环境定义确实是per-agent reward，应该用sum（团队总奖励）
                # 如果用mean会把reward缩小n_agents倍，影响critic尺度和优势函数
                reward = reward.sum(dim=2, keepdim=True)  # (bs, ts, n_agents) -> (bs, ts, 1)
        else:
            # 其他维度，先flatten再处理
            import warnings
            warnings.warn(
                f"Reward shape {reward_original_shape} is unexpected. "
                f"Will flatten and sum to get global reward. "
                f"This may not match your intended high-level reward semantics!",
                UserWarning
            )
            # 使用sum而不是mean（团队奖励）
            reward = reward.view(reward.shape[0], reward.shape[1], -1).sum(dim=2, keepdim=True)
        
        # 统一terminated形状为(bs, ts, 1) - 使用max（任一agent终止就算终止）
        if terminated.dim() == 2:
            terminated = terminated.unsqueeze(-1)  # (bs, ts) -> (bs, ts, 1)
        elif terminated.dim() == 3:
            if terminated.shape[2] == 1:
                # 已经是(bs, ts, 1)，保持
                pass
            else:
                # 如果是(bs, ts, n_agents)等，使用max（任一agent终止就算终止）
                terminated = terminated.max(dim=2, keepdim=True)[0]  # (bs, ts, n_agents) -> (bs, ts, 1)
        elif terminated.dim() == 1:
            # 如果是(ts,)，需要reshape
            terminated = terminated.unsqueeze(0).unsqueeze(-1)  # 需要根据实际情况调整
        
        # 统一reset形状为(bs, ts, 1) - 使用max（任一agent reset就算reset）
        if reset.dim() == 2:
            reset = reset.unsqueeze(-1)  # (bs, ts) -> (bs, ts, 1)
        elif reset.dim() == 3:
            if reset.shape[2] == 1:
                # 已经是(bs, ts, 1)，保持
                pass
            else:
                # 如果是(bs, ts, n_agents)等，使用max
                reset = reset.max(dim=2, keepdim=True)[0]  # (bs, ts, n_agents) -> (bs, ts, 1)
        
        # 统一mask形状为(bs, ts, 1)
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)  # (bs, ts) -> (bs, ts, 1)
        
        allocs = 1 - batch['entity2task_mask'][:, :, :self.args.n_agents].float()
        mask[:, 1:] = mask[:, 1:] * (1 - reset[:, :-1])
        bs, ts, _ = mask.shape
        t_added = batch['t_added'].reshape(bs, 1, 1).repeat(1, ts, 1)

        # 统一timeout形状（reset和terminated已经是(bs, ts, 1)）
        # 关键修复：clamp_min(0.0)避免负值，因为reset和terminated的聚合方式可能导致reset < terminated
        timeout = (reset - terminated).clamp_min(0.0)  # (bs, ts, 1)
        
        # 统一decision_points形状为(bs, ts, 1)
        decision_points = batch['hier_decision'].float()
        if decision_points.dim() == 2:
            decision_points = decision_points.unsqueeze(-1)  # (bs, ts) -> (bs, ts, 1)
        elif decision_points.dim() == 3 and decision_points.shape[2] != 1:
            # 如果已经是3维但最后一维不是1，可能需要处理
            pass
        
        # k-step折扣累积奖励：r_t^H = sum_{i=0}^{k-1} γ^i r_{t+i}
        # 其中 k = action_length, γ = gamma
        # 这是A2C上层奖励的标准实现方式（推荐起步方法）
        k = self.args.hier_agent['action_length']
        gamma = self.args.gamma

        # 现在所有张量都是(bs, ts, 1)，可以安全地使用zeros_like
        seg_rewards = th.zeros_like(reward)  # (bs, ts, 1)
        seg_terminated = th.zeros_like(terminated)  # (bs, ts, 1)
        
        # 简化：现在terminated是(bs, ts, 1)，可以直接取[:, 0]得到(bs, 1)，然后squeeze得到(bs,)
        cuml_terminated = th.zeros(bs, device=terminated.device, dtype=terminated.dtype)  # (bs,)
        cuml_timeout = th.zeros(bs, device=timeout.device, dtype=timeout.dtype)  # (bs,)

        # 计算k-step折扣累积奖励（正确处理终止）
        # 对于每个决策点t，计算从t开始的k-step折扣累积奖励
        # 关键：如果episode在k-step内终止，要停止累积并标记bootstrap mask
        # 现在所有张量都是(bs, ts, 1)，可以简化处理
        bs, ts = reward.shape[:2]
        
        for t in range(ts):
            # 检查哪些batch在时间t是决策点
            is_decision_point = decision_points[:, t, 0] > 0.5  # (bs,)
            
            if is_decision_point.any():
                # 计算从t开始的k-step折扣累积奖励
                # r_t^H = sum_{i=0}^{k-1} γ^i r_{t+i}
                # 关键：如果episode在t+i处终止，停止累积
                discounted_sum = th.zeros(bs, 1, device=reward.device, dtype=reward.dtype)
                discount_factor = 1.0
                alive_mask = th.ones(bs, 1, device=reward.device, dtype=reward.dtype)  # 标记是否还活着
                
                for i in range(k):
                    if t + i < ts:
                        # 检查是否在t+i处终止（terminated已经是(bs, ts, 1)）
                        terminated_at_i = terminated[:, t + i]  # (bs, 1)
                        
                        # 关键修复：先加reward（使用当前的alive_mask），再更新alive_mask
                        # 这样如果终止那一步仍然有reward，不会被alive_mask乘0丢失
                        # 标准做法：终止步的reward应该被包含，但终止后不再累积
                        
                        # reward已经是(bs, ts, 1)，直接取[:, t+i]得到(bs, 1)
                        reward_t_i = reward[:, t + i]  # (bs, 1)
                        
                        # 先加reward（使用当前的alive_mask，包含终止步的reward）
                        discounted_sum = discounted_sum + (discount_factor * reward_t_i * alive_mask)
                        discount_factor *= gamma
                        
                        # 然后更新alive_mask（终止后不再累积下一步的reward）
                        alive_mask = alive_mask * (1 - terminated_at_i)
                        
                        # 如果所有batch都终止了，提前退出
                        if alive_mask.sum() == 0:
                            break
                    else:
                        break
                
                # 只更新决策点的奖励（非决策点保持为0）
                # seg_rewards[:, t]是(bs, 1)，discounted_sum是(bs, 1)，形状匹配
                seg_rewards[:, t] = th.where(
                    is_decision_point.unsqueeze(-1),
                    discounted_sum,
                    seg_rewards[:, t]
                )

        # 从后往前处理，用于计算terminated和timeout
        # 关键：如果在k-step内终止，seg_terminated应该标记为1（不能bootstrap）
        # 关键修复：按is_dp对每个episode单独更新，不要用.any()控制整批分支
        # 问题：.any()会导致只要batch里有任意一个episode在t是决策点，就给所有episode赋值
        # 修复：使用th.where按is_dp逐个episode更新
        for t in reversed(range(ts)):
            # track whether env terminated between decision points
            # 检查哪些episode在时间t是决策点
            is_dp = decision_points[:, t, 0] > 0.5  # (bs,) - 每个episode是否是决策点
            
            if is_dp.any():
                # 对于决策点，检查k-step内是否终止
                # 只对是决策点的episode计算terminated_within_k
                terminated_within_k = th.zeros(bs, 1, device=terminated.device, dtype=terminated.dtype)
                for i in range(k):
                    if t + i < ts:
                        # terminated已经是(bs, ts, 1)，直接取[:, t+i]得到(bs, 1)
                        term_t_i = terminated[:, t + i]  # (bs, 1)
                        # 关键修复：使用th.maximum做element-wise max，而不是.max()（会返回标量）
                        terminated_within_k = th.maximum(terminated_within_k, term_t_i)  # (bs, 1) element-wise max
                    else:
                        break
                
                # 关键修复：只对是决策点的episode更新seg_terminated
                # 使用th.where按is_dp逐个episode更新，避免污染非决策点的episode
                seg_terminated[:, t] = th.where(
                    is_dp.unsqueeze(-1),
                    terminated_within_k,
                    seg_terminated[:, t]
                )
            
            # 非决策点，使用累积终止状态
            # 只对非决策点的episode更新cuml_terminated
            is_not_dp = ~is_dp  # (bs,)
            if is_not_dp.any():
                # terminated已经是(bs, ts, 1)，取[:, t]得到(bs, 1)，然后squeeze得到(bs,)
                term_t = terminated[:, t].squeeze(-1)  # (bs, 1) -> (bs,)
                # 关键修复：使用th.maximum做element-wise max，而不是.max()（会返回标量）
                cuml_terminated = th.maximum(cuml_terminated, term_t)  # (bs,) element-wise max
                
                # 只对非决策点的episode更新seg_terminated
                seg_terminated[:, t, 0] = th.where(
                    is_not_dp,
                    cuml_terminated,
                    seg_terminated[:, t, 0]
                )
                
                # 更新cuml_terminated（只在非决策点累积）
                dp_t = decision_points[:, t, 0].squeeze(-1)  # (bs,)
                cuml_terminated = cuml_terminated * (1 - dp_t)

            # mask out decision point if a env timeout happens (since we can't bootstrap from next decision point)
            # timeout已经是(bs, ts, 1)，取[:, t]得到(bs, 1)，然后squeeze得到(bs,)
            timeout_t = timeout[:, t].squeeze(-1)  # (bs, 1) -> (bs,)
            # 关键修复：使用th.maximum做element-wise max
            cuml_timeout = th.maximum(cuml_timeout, timeout_t)  # (bs,) element-wise max
            
            # 更新mask（timeout时不能bootstrap）
            # mask是(bs, ts, 1)，直接操作
            mask[:, t, 0] *= (1 - cuml_timeout)
            
            # 更新cuml_timeout（只在非决策点累积）
            dp_t = decision_points[:, t, 0].squeeze(-1) if decision_points.dim() == 3 else decision_points[:, t]
            cuml_timeout = cuml_timeout * (1 - dp_t)
        
        # 注意：不再除以action_length，因为k-step折扣累积奖励已经考虑了时间跨度
        # 原来的实现除以action_length是为了归一化，但使用折扣累积后不需要
        # k-step折扣累积奖励已经自然地考虑了时间跨度，不需要额外归一化

        last_alloc = th.zeros_like(allocs)
        was_reset = th.zeros_like(reset[:, [0]])
        for t in range(1, reward.shape[1]):
            # make sure that last_alloc doesn't copy final assignment from previous episode
            was_reset = (was_reset + reset[:, [t - 1]]).min(th.ones_like(was_reset))
            last_alloc[:, t] = allocs[:, t - 1] * (1 - was_reset)
            was_reset *= (1 - decision_points[:, [t]])

        # mask out last decision point in each trajectory if not terminal state (since we can't bootstrap)
        # 风险点2评估：这个处理可能过严，但通常是合理的
        # - 如果最后一个决策点不是terminal，我们无法计算TD target（没有next state），所以mask掉是合理的
        # - 这会让有效样本变少，但避免了使用错误的TD target（bootstrap from nothing）
        # - 如果你的轨迹最后一个决策点后面还有transition但因为截断/采样导致没法bootstrap，这样处理是合理的
        # - 如果确实需要更多样本，可以考虑：只mask掉value loss，但保留policy loss（使用MC return）
        bs, ts, _ = decision_points.shape
        last_dp_ind = (
            decision_points * th.arange(
                ts, dtype=decision_points.dtype,
                device=decision_points.device).reshape(1, ts, 1)
        ).squeeze().argmax(dim=1)
        # 如果最后一个决策点不是terminal，mask掉（无法bootstrap）
        # mask和seg_terminated都是(bs, ts, 1)，需要索引到正确的维度
        mask[th.arange(bs), last_dp_ind, 0] *= seg_terminated[th.arange(bs), last_dp_ind, 0]

        entity2task_mask = batch['entity2task_mask'].clone()

        # 关键修复：正确提取决策点，并保存ep_id和t_id用于时序对齐
        # 问题：d_inds是布尔索引，扁平化后会打乱时序顺序
        # 解决：按episode分组，在每个episode内按时间顺序提取决策点
        d_inds = (decision_points.squeeze(-1) == 1)  # (bs, ts) - 布尔索引
        
        # 为每个决策点保存(ep_id, t_id)
        ep_ids = []
        t_ids = []
        meta_indices = []
        
        for ep_id in range(bs):
            # 在每个episode内，按时间顺序提取决策点
            ep_decision_ts = th.where(d_inds[ep_id])[0]  # 该episode的所有决策点时间步
            for t_id in ep_decision_ts:
                ep_ids.append(ep_id)
                t_ids.append(t_id.item())
                meta_indices.append((ep_id, t_id.item()))
        
        # 关键修复：优化max_bs截断策略，保证next点尽量成对保留
        # 问题：简单截断可能把某个episode的"当前决策点"保留了，但把它的"下一个决策点"裁掉
        # 解决：按episode维度裁，确保对每个保留的(s_t)尽量也保留(s_{t+1})
        max_bs = self.args.hier_agent['max_bs']
        if len(meta_indices) > max_bs:
            # 策略：按episode分组，优先保留完整的episode（包含成对的决策点）
            # 如果必须截断，尽量保留成对的决策点
            ep_to_indices = {}
            for idx, (ep_id, t_id) in enumerate(meta_indices):
                if ep_id not in ep_to_indices:
                    ep_to_indices[ep_id] = []
                ep_to_indices[ep_id].append((idx, t_id))
            
            # 按episode顺序，尽量保留完整的episode
            selected_indices = []
            for ep_id in sorted(ep_to_indices.keys()):
                ep_indices = sorted(ep_to_indices[ep_id], key=lambda x: x[1])  # 按t_id排序
                
                # 尝试保留这个episode的所有决策点
                if len(selected_indices) + len(ep_indices) <= max_bs:
                    selected_indices.extend([idx for idx, _ in ep_indices])
                else:
                    # 如果放不下，至少保留成对的决策点（尽量保留连续的决策点对）
                    remaining = max_bs - len(selected_indices)
                    if remaining >= 2:
                        # 保留前remaining个（尽量成对）
                        selected_indices.extend([idx for idx, _ in ep_indices[:remaining]])
                    break
            
            # 按原始顺序排序（保持时序）
            selected_indices = sorted(selected_indices)
            meta_indices = [meta_indices[i] for i in selected_indices]
            ep_ids = [ep_ids[i] for i in selected_indices]
            t_ids = [t_ids[i] for i in selected_indices]
        
        # 使用保存的索引提取数据（保持时序顺序）
        meta_batch = {}
        for key in ['reward', 'terminated', 'mask', 'entities', 'obs_mask', 
                    'entity_mask', 'entity2task_mask', 'task_mask', 'avail_actions', 
                    'last_alloc', 't_added', 'old_log_prob', 'old_actions', 'alloc_order']:
            if key == 'reward':
                source = seg_rewards
            elif key == 'terminated':
                source = seg_terminated
            elif key == 'mask':
                source = mask
            elif key == 'entity2task_mask':
                source = entity2task_mask
            elif key == 'last_alloc':
                source = last_alloc
            elif key == 't_added':
                source = t_added
            elif key in ['old_log_prob', 'old_actions', 'alloc_order']:
                # 这些字段只在PPO时需要，从batch中提取
                if key in batch.data.transition_data:
                    source = batch.data.transition_data[key]
                else:
                    # 如果不存在，创建占位符（不应该发生）
                    if key == 'old_log_prob':
                        source = th.zeros((batch.batch_size, batch.max_seq_length, 1), 
                                         device=batch.device, dtype=th.float32)
                    elif key == 'old_actions':
                        source = th.zeros((batch.batch_size, batch.max_seq_length, 
                                          self.args.n_agents, self.args.n_tasks),
                                         device=batch.device, dtype=th.float32)
                    else:  # alloc_order
                        # 固定顺序：0..na-1
                        source = th.arange(self.args.n_agents, device=batch.device, dtype=th.long).unsqueeze(0).unsqueeze(0).repeat(
                            batch.batch_size, batch.max_seq_length, 1
                        )  # (bs, max_seq, na)
            else:
                source = batch[key]
            
            # 按(ep_id, t_id)提取，保持时序顺序
            extracted = []
            for ep_id, t_id in meta_indices:
                extracted.append(source[ep_id, t_id])
            meta_batch[key] = th.stack(extracted, dim=0)
        
        # 保存ep_id和t_id用于A2C训练时的next_state对齐
        meta_batch['ep_id'] = th.tensor(ep_ids, device=batch.device, dtype=th.long)
        meta_batch['t_id'] = th.tensor(t_ids, device=batch.device, dtype=th.long)
        
        return meta_batch

    def alloc_train_aql(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        meta_batch = self._make_meta_batch(batch)
        rewards = meta_batch['reward']
        terminated = meta_batch['terminated']
        mask = meta_batch['mask']
        stats = {}

        # Compute Q-values (evaluate task allocation stored in entity2task_mask)
        alloc_q, q_stats = self.mac.evaluate_allocation(meta_batch, calc_stats=True)

        # Compute proposal allocations (test_mode=True to remove stochasticity in critic, pass in target_mac for stability in bootstrap targets)
        new_alloc, pi_stats = self.mac.compute_allocation(meta_batch, calc_stats=True, test_mode=True, target_mac=self.target_mac)

        # Compute target Q-values
        target_alloc_q = pi_stats['targ_best_prop_values']
        target_alloc_q = self.target_mac.alloc_critic.denormalize(target_alloc_q)

        # Compute TD-loss (don't bootstrap from next state if previous state is
        # terminal)
        targets = (rewards[:-1] + self.args.gamma * (1 - terminated[:-1]) * target_alloc_q[1:]).detach()
        if self.args.popart:
            targets = self.mac.alloc_critic.popart_update(
                targets, mask[:-1])

        td_error = (alloc_q[:-1] - targets.detach())
        td_mask = mask[:-1].expand_as(td_error)
        if self.args.hier_agent['decay_old'] > 0:
            cutoff = self.args.hier_agent['decay_old']
            ratio = (cutoff - t_env + meta_batch['t_added'][:-1].float()) / cutoff
            ratio = ratio.max(th.zeros_like(ratio))
            td_mask *= ratio
        masked_td_error = td_error * td_mask
        # Normal L2 loss, take mean over actual data
        td_loss = (masked_td_error ** 2).sum() / td_mask.sum()
        stats['losses/alloc_q_loss'] = td_loss.cpu().item()

        # backprop Q loss
        q_loss = td_loss
        self.alloc_q_optimiser.zero_grad()
        q_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.alloc_q_params, self.args.grad_norm_clip)
        stats['train_metrics/alloc_q_grad_norm'] = grad_norm
        self.alloc_q_optimiser.step()

        # Log allocation metrics
        stats['alloc_metrics/best_prob'] = pi_stats['best_prob'].mean().cpu().item()
        # Compute what % of agents changed their task allocation (if previous alloc exists)
        active_ag = 1 - meta_batch['entity_mask'][:, :self.args.n_agents].float()
        ag_changed = (meta_batch['last_alloc'].argmax(dim=2) != new_alloc.detach().argmax(dim=2)).float()
        prev_al_exists = (meta_batch['last_alloc'].sum(dim=(1, 2)) >= 1).float()
        perc_changed_per_step = ((ag_changed * active_ag).sum(dim=1) / active_ag.sum(dim=1))
        perc_changed = (perc_changed_per_step * prev_al_exists).sum() / prev_al_exists.sum()
        stats['alloc_metrics/perc_ag_changed'] = perc_changed.cpu().item()
        # Measure abs value of difference between # of agents and # of entities
        # in each subtask (may not be useful for all tasks)
        nonagent2task = 1 - meta_batch['entity2task_mask'][:, self.args.n_agents:].float()
        ag_per_task = new_alloc.detach().sum(dim=1)
        nag_per_task = nonagent2task.sum(dim=1)
        absdiff_per_task = (ag_per_task - nag_per_task).abs()
        abs_diff_mean = absdiff_per_task.sum(dim=1) / (1 - meta_batch['task_mask'].float()).sum(dim=1)
        stats['alloc_metrics/ag_task_concentration'] = abs_diff_mean.mean().cpu().item()

        # Maximize probability of best allocation
        all_prop_log_pi = pi_stats['log_pi']  # log_pi of all sampled proposal actions
        bs = all_prop_log_pi.shape[0]
        best_prop_log_pi = all_prop_log_pi[th.arange(bs), pi_stats['best_prop_inds']]
        amort_step_loss = -best_prop_log_pi
        masked_amort_step_loss = amort_step_loss * mask
        amort_loss = masked_amort_step_loss.sum() / mask.sum()
        stats['losses/alloc_amort_loss'] = amort_loss.cpu().item()


        active_task = 1 - meta_batch['task_mask'].float().unsqueeze(1)
        ag2task = pi_stats['all_allocs'].detach()  # (bs, n_prop, na, nt)
        task_has_agents = (ag2task.sum(dim=2) > 0).float()
        any_task_no_agents = (task_has_agents.sum(dim=2, keepdim=True)
                              != active_task.sum(dim=2, keepdim=True)).float()
        stats['alloc_metrics/any_task_no_agents_pi'] = any_task_no_agents.mean().cpu().item()

        # entropy term
        entropy = pi_stats['entropy']
        entropy_loss = -entropy.mean()
        stats['losses/alloc_entropy'] = -entropy_loss.cpu().item()

        pi_loss = (amort_loss
                   + self.args.hier_agent['entropy_loss'] * entropy_loss)

        # backprop policy loss
        self.alloc_pi_optimiser.zero_grad()
        pi_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.alloc_pi_params, self.args.grad_norm_clip)
        stats['train_metrics/alloc_pi_grad_norm'] = grad_norm
        self.alloc_pi_optimiser.step()

        if (episode_num - self.last_alloc_target_update_episode) / self.args.alloc_target_update_interval >= 1.0:
            self._update_alloc_targets()
            self.last_alloc_target_update_episode = episode_num

        if t_env - self.log_alloc_stats_t >= self.args.learner_log_interval:
            for name, value in stats.items():
                self.logger.log_stat(name, value, t_env)
            self.log_alloc_stats_t = t_env

        return stats, new_alloc

    def alloc_train_a2c(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        """
        A2C算法训练任务分配策略
        
        训练流程：
        1. Actor采样动作
        2. Critic评估V值
        3. 计算优势函数 A = Q - V = r + γV' - V
        4. Actor使用策略梯度更新
        5. Critic使用TD误差更新
        """
        meta_batch = self._make_meta_batch(batch)
        rewards = meta_batch['reward']
        terminated = meta_batch['terminated']
        mask = meta_batch['mask']
        stats = {}
        
        # 1. Actor采样动作（训练模式）
        self.mac.alloc_policy.train()
        actions, pi_stats = self.mac.compute_allocation(
            meta_batch, calc_stats=True, test_mode=False
        )
        log_probs = pi_stats['log_prob']  # (bs, 1) - 联合log概率
        entropy = pi_stats['entropy']      # (bs, 1) - 联合熵
        
        # 2. Critic评估当前状态V值
        values = self.mac.evaluate_allocation(meta_batch)  # (bs, 1)
        
        # 检查是否有NaN
        if th.isnan(values).any():
            print(f"Warning: NaN detected in current values at t_env={t_env}")
            # 使用零值替换NaN
            values = th.where(th.isnan(values), th.zeros_like(values), values)
        
        # 3. 构建下一状态batch（关键修复：确保next_state是同一条trajectory的下一个决策点）
        # 问题：meta_batch[1:]不是真正的next state，而是扁平序列的下一个元素
        # 修复：使用ep_id和t_id，为每个meta transition找到同一条episode的下一个决策点
        bs = meta_batch['entities'].shape[0]
        next_indices = []  # 初始化，确保在所有分支中都有定义
        valid_next_mask = None  # 初始化
        
        if bs <= 1:
            # 如果只有1个样本，无法构建下一状态
            next_meta_batch = {}
            required_keys = ['entities', 'entity_mask', 'entity2task_mask', 'task_mask', 'obs_mask', 'last_alloc']
            for key in required_keys:
                if key in meta_batch:
                    if len(meta_batch[key].shape) > 0:
                        next_meta_batch[key] = meta_batch[key][:0]  # 空tensor
                    else:
                        next_meta_batch[key] = meta_batch[key]
            # 没有有效的next_state
            valid_next_mask = th.zeros(max(0, bs - 1), device=meta_batch['entities'].device, dtype=th.bool)
        else:
            # 关键修复：使用ep_id和t_id正确对齐next_state
            # 为每个meta transition找到同一条episode的下一个决策点
            ep_ids = meta_batch['ep_id']  # (bs,)
            t_ids = meta_batch['t_id']    # (bs,)
            
            # 构建next_state索引：对每个当前transition，找到同ep的下一个决策点
            next_indices = []
            for i in range(bs - 1):  # 最后一个没有next
                current_ep_id = ep_ids[i].item()
                current_t_id = t_ids[i].item()
                
                # 在同一条episode内，找到下一个决策点（t_id > current_t_id且最小）
                next_idx = None
                for j in range(i + 1, bs):
                    if ep_ids[j].item() == current_ep_id and t_ids[j].item() > current_t_id:
                        next_idx = j
                        break
                
                next_indices.append(next_idx)
            
            # 构建next_meta_batch：只包含有next_state的transition
            valid_mask = th.tensor([idx is not None for idx in next_indices], 
                                  device=meta_batch['entities'].device, dtype=th.bool)
            
            if valid_mask.sum() == 0:
                # 没有有效的next_state，创建空batch
                next_meta_batch = {}
                required_keys = ['entities', 'entity_mask', 'entity2task_mask', 'task_mask', 'obs_mask', 'last_alloc']
                for key in required_keys:
                    if key in meta_batch:
                        if len(meta_batch[key].shape) > 0:
                            next_meta_batch[key] = meta_batch[key][:0]
                        else:
                            next_meta_batch[key] = meta_batch[key]
            else:
                # 提取有效的next_state
                next_indices_tensor = th.tensor([idx for idx in next_indices if idx is not None],
                                               device=meta_batch['entities'].device, dtype=th.long)
                
                required_keys = ['entities', 'entity_mask', 'entity2task_mask', 'task_mask', 'obs_mask', 'last_alloc']
                next_meta_batch = {}
                for key in required_keys:
                    if key in meta_batch:
                        next_meta_batch[key] = meta_batch[key][next_indices_tensor]
                
                # 可选键
                optional_keys = ['avail_actions']
                for key in optional_keys:
                    if key in meta_batch:
                        next_meta_batch[key] = meta_batch[key][next_indices_tensor]
        
        # 4. Critic评估下一状态V值（使用目标网络）
        with th.no_grad():
            # 如果batch size <= 1或没有有效的next_state，创建空的next_values
            if bs <= 1 or (bs > 1 and 'entities' in next_meta_batch and next_meta_batch['entities'].shape[0] == 0):
                # 创建空的next_values，shape为(0, 1)
                next_values = th.zeros((0, 1), device=values.device, dtype=values.dtype)
                valid_next_mask = th.zeros(bs - 1, device=values.device, dtype=th.bool)
            else:
                # 下一状态的V值（使用构建的next_meta_batch）
                next_values = self.target_mac.evaluate_allocation(next_meta_batch)
                
                # 检查是否有NaN
                if th.isnan(next_values).any():
                    print(f"Warning: NaN detected in next values at t_env={t_env}")
                    # 使用零值替换NaN
                    next_values = th.where(th.isnan(next_values), th.zeros_like(next_values), next_values)
                
                # 关键修复：删除denormalize，保持values和next_values在同一尺度空间
                # 如果critic输出是normalized，那就都用normalized；如果是raw，那就都用raw
                # 不要混用denormalize，这会导致TD error尺度不一致
                # 先关闭PopArt或统一尺度，等A2C能学起来后再正确接入PopArt
                # next_values = self.target_mac.alloc_critic.denormalize(next_values)  # 已删除
                
                # 构建valid_next_mask：标记哪些transition有有效的next_state
                valid_next_mask = th.tensor([idx is not None for idx in next_indices],
                                           device=values.device, dtype=th.bool)
        
        # 5. 计算TD目标和优势函数
        # TD目标：r + γ^k * V(s')（关键修复：使用γ^k而不是γ）
        # 因为高层动作持续k步，bootstrap时需要用γ^k
        # 关键修复：不丢弃最后一个决策点！对于没有next_state的transition，使用 r 作为TD target
        k = self.args.hier_agent['action_length']
        gamma_k = self.args.gamma ** k  # γ^k
        
        # 当前状态V值（使用所有transition，包括最后一个）
        current_values = values  # (bs, 1)
        
        # 构建TD targets：包括所有transition
        # 对于有next_state的：r + γ^k * V(s')
        # 对于没有next_state的（最后一个决策点）：r（terminal state的value就是reward）
        td_targets = th.zeros_like(current_values)  # (bs, 1)
        
        # 处理有next_state的transition（前bs-1个）
        # 关键修复1：使用显式索引，避免链式索引写不回的问题
        if bs > 1 and valid_next_mask.sum() > 0:
            # 获取有效样本的显式索引（对应0..bs-2的位置）
            idx = th.nonzero(valid_next_mask, as_tuple=False).squeeze(-1)  # shape (n_valid,)
            
            # 提取有效的数据
            valid_current_values = current_values[idx]  # (n_valid, 1)
            valid_rewards = rewards[idx]  # (n_valid, 1)
            valid_terminated = terminated[idx]  # (n_valid, 1)
            valid_next_values = next_values  # (n_valid, 1)
            
            # 关键修复3：显式对齐检查（调试期非常有用）
            assert next_values.shape[0] == idx.numel(), \
                f"next_values shape {next_values.shape[0]} != valid indices {idx.numel()}"
            
            # 计算TD target（只对有效的transition）
            td_targets_valid = (
                valid_rewards + 
                gamma_k * (1 - valid_terminated) * valid_next_values
            ).detach()
            
            # 关键修复1：使用显式索引写回，确保赋值生效
            td_targets[idx] = td_targets_valid
        
        # 关键修复6：处理所有没有next_state的transition（不只是最后一个）
        # 对于terminal state，TD target就是reward本身
        # 找出所有"没有next"的样本位置
        no_next_mask = th.zeros(bs, dtype=th.bool, device=values.device)
        if bs > 1:
            # 前bs-1个样本：没有next的标记为True
            no_next_mask[:-1] = ~valid_next_mask
        # 最后一个样本必然没有next
        no_next_mask[-1] = True
        
        # 只对有效的样本（有mask且没有next）设置TD target为reward
        # 关键修复2：正确处理mask形状，避免tensor判断问题
        # mask是(bs, 1)，需要squeeze或直接比较
        mask_flat = mask.squeeze(-1)  # (bs,)
        valid_no_next = no_next_mask & (mask_flat > 0.0)  # (bs,)
        if valid_no_next.any():
            td_targets[valid_no_next] = rewards[valid_no_next].detach()  # terminal state: V(s) = r
        
        # 关键修复：先关闭PopArt，统一尺度
        # PopArt会导致尺度不一致：values可能是normalized，但td_targets被denormalize
        # 等A2C能学起来（成功率明显上升）后，再回来把PopArt按一致尺度正确接入
        # if self.args.popart:
        #     td_targets = self.mac.alloc_critic.popart_update(
        #         td_targets, mask
        #     )
        
        # 优势函数：A = Q - V = r + γ^k*V' - V（或 r - V 对于terminal state）
        advantages = td_targets - current_values  # (bs, 1)
        
        # 5. Critic损失（TD误差）
        td_error = (current_values - td_targets.detach())  # (bs, 1)
        
        # 关键修复：使用所有有效的transition（包括最后一个决策点）
        # mask是(bs, 1)，直接使用
        td_mask = mask  # (bs, 1)
        
        # 处理旧数据衰减（如果启用）
        if self.args.hier_agent.get('decay_old', 0) > 0:
            cutoff = self.args.hier_agent['decay_old']
            ratio = (cutoff - t_env + meta_batch['t_added'].float()) / cutoff
            ratio = ratio.max(th.zeros_like(ratio))
            # ratio是(bs, 1)，直接相乘
            td_mask = td_mask * ratio
        
        # 关键修复：避免除以0导致NaN
        # 如果td_mask全为0（比如所有样本都没有有效next_state），会导致NaN
        den = td_mask.sum().clamp_min(1e-8)
        value_loss = (td_error ** 2 * td_mask).sum() / den
        stats['losses/alloc_value_loss'] = value_loss.cpu().item()
        
        # 更新Critic
        self.alloc_q_optimiser.zero_grad()
        value_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.alloc_q_params, 
            self.args.grad_norm_clip
        )
        stats['train_metrics/alloc_value_grad_norm'] = grad_norm
        
        # 检查梯度是否有NaN
        has_nan_grad = False
        for param in self.alloc_q_params:
            if param.grad is not None and th.isnan(param.grad).any():
                has_nan_grad = True
                if not hasattr(self, '_nan_grad_warning_count_critic'):
                    self._nan_grad_warning_count_critic = 0
                self._nan_grad_warning_count_critic += 1
                if self._nan_grad_warning_count_critic <= 1 or self._nan_grad_warning_count_critic % 100 == 0:
                    print(f"Warning: NaN gradient detected in Critic params (count: {self._nan_grad_warning_count_critic}). Skipping update.")
                break
        
        if not has_nan_grad:
            self.alloc_q_optimiser.step()
            
            # 检查参数是否有NaN（更新后）
            has_nan_param = False
            for param in self.alloc_q_params:
                if th.isnan(param.data).any():
                    has_nan_param = True
                    if not hasattr(self, '_nan_param_warning_count_critic'):
                        self._nan_param_warning_count_critic = 0
                    self._nan_param_warning_count_critic += 1
                    if self._nan_param_warning_count_critic <= 1 or self._nan_param_warning_count_critic % 100 == 0:
                        print(f"ERROR: NaN in Critic parameters after update (count: {self._nan_param_warning_count_critic})! This is serious.")
                    # 重置NaN参数为零
                    param.data = th.where(th.isnan(param.data), th.zeros_like(param.data), param.data)
                    break
        
        # 6. Actor损失（策略梯度）
        # 使用优势函数加权log概率
        # 注意：log_probs是联合log概率（所有智能体的log概率之和）
        # 关键修复：去掉除以n_agents的归一化，因为team reward + per-agent log_prob sum的设定下不需要
        # 并添加优势归一化来控尺度（非常管用，且不改变期望梯度方向）
        
        # 准备优势函数和mask
        adv_used = advantages.detach()  # (bs, 1)
        policy_mask = td_mask  # (bs, 1) - 包含所有有效的transition（包括最后一个决策点）
        
        # 优势归一化：减去均值，除以标准差
        # 这可以稳定训练，且不改变期望梯度方向
        adv_mean = (adv_used * policy_mask).sum() / (policy_mask.sum() + 1e-8)
        adv_var = (((adv_used - adv_mean) ** 2) * policy_mask).sum() / (policy_mask.sum() + 1e-8)
        adv_norm = (adv_used - adv_mean) / th.sqrt(adv_var + 1e-8)
        
        # 关键修复：去掉除以n_agents，直接使用log_probs
        # 使用所有有效的transition（包括最后一个决策点）
        # log_probs: (bs, 1)
        # adv_norm: (bs, 1)
        # policy_mask: (bs, 1)
        policy_loss = -(
            log_probs * adv_norm * policy_mask
        ).sum() / (policy_mask.sum() + 1e-8)  # 避免除零
        
        # 熵正则化（使用所有有效的transition）
        entropy_loss = -entropy.mean()
        
        pi_loss = (
            policy_loss + 
            self.args.hier_agent.get('entropy_loss', 0.01) * entropy_loss
        )
        
        stats['losses/alloc_policy_loss'] = policy_loss.cpu().item()
        stats['losses/alloc_entropy'] = -entropy_loss.cpu().item()
        stats['losses/alloc_total_loss'] = pi_loss.cpu().item()
        
        # 优势函数统计
        # 关键修复：保持policy_mask形状一致，不要squeeze
        # 统计归一化前后的优势函数
        advantages_masked = advantages * policy_mask
        advantages_valid = advantages_masked[policy_mask > 0]
        advantages_norm_masked = adv_norm * policy_mask
        advantages_norm_valid = advantages_norm_masked[policy_mask > 0]
        if len(advantages_valid) > 0:
            stats['train_metrics/alloc_advantage_mean'] = advantages_valid.mean().cpu().item()
            stats['train_metrics/alloc_advantage_std'] = advantages_valid.std().cpu().item()
            stats['train_metrics/alloc_advantage_max'] = advantages_valid.max().cpu().item()
            stats['train_metrics/alloc_advantage_min'] = advantages_valid.min().cpu().item()
            stats['train_metrics/alloc_advantage_abs_mean'] = advantages_valid.abs().mean().cpu().item()
            # 归一化后的优势函数统计
            if len(advantages_norm_valid) > 0:
                stats['train_metrics/alloc_advantage_norm_mean'] = advantages_norm_valid.mean().cpu().item()
                stats['train_metrics/alloc_advantage_norm_std'] = advantages_norm_valid.std().cpu().item()
        else:
            stats['train_metrics/alloc_advantage_mean'] = 0.0
            stats['train_metrics/alloc_advantage_std'] = 0.0
            stats['train_metrics/alloc_advantage_max'] = 0.0
            stats['train_metrics/alloc_advantage_min'] = 0.0
            stats['train_metrics/alloc_advantage_abs_mean'] = 0.0
            stats['train_metrics/alloc_advantage_norm_mean'] = 0.0
            stats['train_metrics/alloc_advantage_norm_std'] = 0.0
        
        # TD误差统计
        # td_mask已经是(bs, 1)，直接使用（包含所有有效的transition，包括最后一个决策点）
        td_error_masked = td_error * td_mask
        td_error_valid = td_error_masked[td_mask > 0]
        if len(td_error_valid) > 0:
            stats['train_metrics/alloc_td_error_mean'] = td_error_valid.mean().cpu().item()
            stats['train_metrics/alloc_td_error_std'] = td_error_valid.std().cpu().item()
            stats['train_metrics/alloc_td_error_abs_mean'] = td_error_valid.abs().mean().cpu().item()
        else:
            stats['train_metrics/alloc_td_error_mean'] = 0.0
            stats['train_metrics/alloc_td_error_std'] = 0.0
            stats['train_metrics/alloc_td_error_abs_mean'] = 0.0
        
        # 价值函数统计
        values_masked = current_values * td_mask
        values_valid = values_masked[td_mask > 0]
        td_targets_masked = td_targets.detach() * td_mask
        td_targets_valid = td_targets_masked[td_mask > 0]
        if len(values_valid) > 0:
            stats['train_metrics/alloc_value_mean'] = values_valid.mean().cpu().item()
            stats['train_metrics/alloc_value_std'] = values_valid.std().cpu().item()
            stats['train_metrics/alloc_value_max'] = values_valid.max().cpu().item()
            stats['train_metrics/alloc_value_min'] = values_valid.min().cpu().item()
        if len(td_targets_valid) > 0:
            stats['train_metrics/alloc_td_target_mean'] = td_targets_valid.mean().cpu().item()
            stats['train_metrics/alloc_td_target_std'] = td_targets_valid.std().cpu().item()
        
        # 策略统计
        # 关键修复：保持policy_mask形状一致，不要squeeze
        # 使用所有有效的transition（包括最后一个决策点）
        log_probs_masked = log_probs * policy_mask
        log_probs_valid = log_probs_masked[policy_mask > 0]
        entropy_masked = entropy * policy_mask
        entropy_valid = entropy_masked[policy_mask > 0]
        if len(log_probs_valid) > 0:
            stats['train_metrics/alloc_log_prob_mean'] = log_probs_valid.mean().cpu().item()
            stats['train_metrics/alloc_log_prob_std'] = log_probs_valid.std().cpu().item()
        if len(entropy_valid) > 0:
            stats['train_metrics/alloc_entropy_mean'] = entropy_valid.mean().cpu().item()
            stats['train_metrics/alloc_entropy_std'] = entropy_valid.std().cpu().item()
            stats['train_metrics/alloc_entropy_max'] = entropy_valid.max().cpu().item()
            stats['train_metrics/alloc_entropy_min'] = entropy_valid.min().cpu().item()
        
        # 奖励统计
        # 关键修复：保持policy_mask形状一致，不要squeeze
        rewards_masked = rewards * policy_mask
        rewards_valid = rewards_masked[policy_mask > 0]
        if len(rewards_valid) > 0:
            stats['train_metrics/alloc_reward_mean'] = rewards_valid.mean().cpu().item()
            stats['train_metrics/alloc_reward_std'] = rewards_valid.std().cpu().item()
            stats['train_metrics/alloc_reward_sum'] = rewards_valid.sum().cpu().item()
        
        # 动作分布统计（从actions中提取）
        if actions is not None:
            # actions: (bs, na, nt) - one-hot编码
            # 计算每个智能体分配到的任务索引
            action_indices = actions.argmax(dim=-1)  # (bs, na)
            # 统计任务分配分布
            na = self.args.n_agents
            nt = self.args.n_tasks
            task_distribution = th.zeros(nt, device=actions.device)
            for t in range(nt):
                task_distribution[t] = (action_indices == t).sum().float()
            task_distribution = task_distribution / (action_indices.numel() + 1e-8)  # 归一化
            stats['alloc_metrics/task_distribution_entropy'] = -(task_distribution * (task_distribution + 1e-8).log()).sum().cpu().item()
            stats['alloc_metrics/task_distribution_max'] = task_distribution.max().cpu().item()
            stats['alloc_metrics/task_distribution_min'] = task_distribution.min().cpu().item()
        
        # 更新Actor
        self.alloc_pi_optimiser.zero_grad()
        pi_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.alloc_pi_params, 
            self.args.grad_norm_clip
        )
        stats['train_metrics/alloc_pi_grad_norm'] = grad_norm
        
        # 检查梯度是否有NaN
        has_nan_grad = False
        for param in self.alloc_pi_params:
            if param.grad is not None and th.isnan(param.grad).any():
                has_nan_grad = True
                if not hasattr(self, '_nan_grad_warning_count'):
                    self._nan_grad_warning_count = 0
                self._nan_grad_warning_count += 1
                if self._nan_grad_warning_count <= 1 or self._nan_grad_warning_count % 100 == 0:
                    print(f"Warning: NaN gradient detected in Actor params (count: {self._nan_grad_warning_count}). Skipping update.")
                break
        
        if not has_nan_grad:
            self.alloc_pi_optimiser.step()
            
            # 检查参数是否有NaN（更新后）
            has_nan_param = False
            for param in self.alloc_pi_params:
                if th.isnan(param.data).any():
                    has_nan_param = True
                    if not hasattr(self, '_nan_param_warning_count'):
                        self._nan_param_warning_count = 0
                    self._nan_param_warning_count += 1
                    if self._nan_param_warning_count <= 1 or self._nan_param_warning_count % 100 == 0:
                        print(f"ERROR: NaN in Actor parameters after update (count: {self._nan_param_warning_count})! This is serious.")
                    # 重置NaN参数为零
                    param.data = th.where(th.isnan(param.data), th.zeros_like(param.data), param.data)
                    break
        
        # 7. 更新目标网络
        if (episode_num - self.last_alloc_target_update_episode) / \
           self.args.alloc_target_update_interval >= 1.0:
            self._update_alloc_targets()
            self.last_alloc_target_update_episode = episode_num
        
        # 8. 添加学习率和探索率指标
        if hasattr(self, 'alloc_pi_optimiser') and len(self.alloc_pi_optimiser.param_groups) > 0:
            stats['train_metrics/alloc_pi_lr'] = self.alloc_pi_optimiser.param_groups[0]['lr']
        if hasattr(self, 'alloc_q_optimiser') and len(self.alloc_q_optimiser.param_groups) > 0:
            stats['train_metrics/alloc_q_lr'] = self.alloc_q_optimiser.param_groups[0]['lr']
        
        # 探索率（从mac中获取）
        if hasattr(self.mac, 'alloc_eps'):
            stats['train_metrics/alloc_epsilon'] = self.mac.alloc_eps
        
        # 批次大小统计
        stats['train_metrics/alloc_batch_size'] = float(td_mask.sum().cpu().item())
        stats['train_metrics/alloc_valid_samples'] = float((td_mask > 0).sum().cpu().item())
        
        # 9. 记录统计信息
        if t_env - self.log_alloc_stats_t >= self.args.learner_log_interval:
            for name, value in stats.items():
                self.logger.log_stat(name, value, t_env)
            self.log_alloc_stats_t = t_env
        
        return stats, actions

    def alloc_train_ppo(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        """
        PPO算法训练任务分配策略
        
        训练流程：
        1. 从batch中提取旧策略的log_prob和actions（固定）
        2. 使用当前策略计算固定actions的log_prob（不重新采样）
        3. 计算重要性采样比率 r(θ) = exp(log_prob_new - log_prob_old)
        4. 计算优势函数 A = TD_target - V(s)（TD residual，不是r + γ^k*V' - V）
        5. 使用clipped objective更新Actor
        6. 使用TD误差更新Critic（每个epoch都更新）
        7. 多个epoch更新（使用同一批数据，minibatch shuffle）
        """
        meta_batch = self._make_meta_batch(batch)
        rewards = meta_batch['reward']
        terminated = meta_batch['terminated']
        mask = meta_batch['mask']
        stats = {}
        
        # 提取旧策略的log_prob和actions（从episode_batch中）
        # 关键修复：确保old_log_probs的shape严格为(bs, 1)
        ep_ids = meta_batch['ep_id']  # (bs,)
        t_ids = meta_batch['t_id']    # (bs,)
        bs = len(ep_ids)
        
        # 如果batch size为0，直接返回（避免后续错误）
        if bs == 0:
            return stats, None
        old_log_probs = []
        old_actions = []
        alloc_orders = []
        for i in range(bs):
            ep_id = ep_ids[i].item()
            t_id = t_ids[i].item()
            if 'old_log_prob' in batch.data.transition_data:
                old_log_prob = batch.data.transition_data['old_log_prob'][ep_id, t_id]  # 可能是(1,)或标量
                # 强制转换为(1,)形状
                if old_log_prob.dim() == 0:
                    old_log_prob = old_log_prob.view(1)
                elif old_log_prob.dim() > 1:
                    old_log_prob = old_log_prob.view(-1)[0:1]  # 取第一个元素并reshape为(1,)
                old_log_probs.append(old_log_prob)
            else:
                old_log_probs.append(th.zeros(1, device=meta_batch['entities'].device))
            
            if 'old_actions' in batch.data.transition_data:
                old_action = batch.data.transition_data['old_actions'][ep_id, t_id]  # (na, nt)
                old_actions.append(old_action)
            else:
                # 如果没有old_actions，说明是第一次训练，需要采样（不应该发生）
                old_actions.append(th.zeros(self.args.n_agents, self.args.n_tasks, device=meta_batch['entities'].device))
            
            # 提取alloc_order（AR模式需要）
            if 'alloc_order' in batch.data.transition_data:
                alloc_order = batch.data.transition_data['alloc_order'][ep_id, t_id]  # (na,) 或 (1, na)
                if alloc_order.dim() == 1:
                    alloc_orders.append(alloc_order)
                else:
                    alloc_orders.append(alloc_order.squeeze(0))
            else:
                # 如果没有alloc_order，使用固定顺序（并行模式）
                alloc_orders.append(th.arange(self.args.n_agents, device=meta_batch['entities'].device))
        
        # 确保old_log_probs严格为(bs, 1)
        old_log_probs = th.stack(old_log_probs, dim=0)  # (bs, 1) 或 (bs,)
        if old_log_probs.dim() == 1:
            old_log_probs = old_log_probs.unsqueeze(-1)  # (bs, 1)
        elif old_log_probs.dim() > 2:
            old_log_probs = old_log_probs.view(bs, -1)[:, 0:1]  # 取第一列并reshape为(bs, 1)
        
        # 确保old_actions为(bs, na, nt)
        old_actions = th.stack(old_actions, dim=0)  # (bs, na, nt)
        
        # 保存alloc_orders到meta_batch（用于AR模式）
        meta_batch['alloc_order'] = th.stack(alloc_orders, dim=0)  # (bs, na)
        
        # PPO超参数
        ppo_epochs = self.args.hier_agent.get('ppo_epochs', 4)
        ppo_clip = self.args.hier_agent.get('ppo_clip', 0.2)
        ppo_minibatch_size = self.args.hier_agent.get('ppo_minibatch_size', None)  # None表示使用full batch
        
        # 1. 计算TD目标和优势函数（在epoch循环外计算一次，detach以便复用）
        self.mac.alloc_policy.train()
        values = self.mac.evaluate_allocation(meta_batch)  # (bs, 1)
        
        if th.isnan(values).any():
            print(f"Warning: NaN detected in current values at t_env={t_env}")
            values = th.where(th.isnan(values), th.zeros_like(values), values)
        
        # 构建下一状态batch
        next_indices = []
        valid_next_mask = None
        
        if bs <= 1:
            next_meta_batch = {}
            required_keys = ['entities', 'entity_mask', 'entity2task_mask', 'task_mask', 'obs_mask', 'last_alloc']
            for key in required_keys:
                if key in meta_batch:
                    if len(meta_batch[key].shape) > 0:
                        next_meta_batch[key] = meta_batch[key][:0]
                    else:
                        next_meta_batch[key] = meta_batch[key]
            valid_next_mask = th.zeros(max(0, bs - 1), device=meta_batch['entities'].device, dtype=th.bool)
        else:
            next_indices = []
            for i in range(bs - 1):
                current_ep_id = ep_ids[i].item()
                current_t_id = t_ids[i].item()
                next_idx = None
                for j in range(i + 1, bs):
                    if ep_ids[j].item() == current_ep_id and t_ids[j].item() > current_t_id:
                        next_idx = j
                        break
                next_indices.append(next_idx)
            
            valid_mask = th.tensor([idx is not None for idx in next_indices], 
                                  device=meta_batch['entities'].device, dtype=th.bool)
            
            if valid_mask.sum() == 0:
                next_meta_batch = {}
                required_keys = ['entities', 'entity_mask', 'entity2task_mask', 'task_mask', 'obs_mask', 'last_alloc']
                for key in required_keys:
                    if key in meta_batch:
                        if len(meta_batch[key].shape) > 0:
                            next_meta_batch[key] = meta_batch[key][:0]
                        else:
                            next_meta_batch[key] = meta_batch[key]
            else:
                next_indices_tensor = th.tensor([idx for idx in next_indices if idx is not None],
                                               device=meta_batch['entities'].device, dtype=th.long)
                required_keys = ['entities', 'entity_mask', 'entity2task_mask', 'task_mask', 'obs_mask', 'last_alloc']
                next_meta_batch = {}
                for key in required_keys:
                    if key in meta_batch:
                        next_meta_batch[key] = meta_batch[key][next_indices_tensor]
                optional_keys = ['avail_actions']
                for key in optional_keys:
                    if key in meta_batch:
                        next_meta_batch[key] = meta_batch[key][next_indices_tensor]
        
        # 计算next_values（关键修复：使用target critic，提高稳定性）
        with th.no_grad():
            if bs <= 1 or (bs > 1 and 'entities' in next_meta_batch and next_meta_batch['entities'].shape[0] == 0):
                next_values = th.zeros((0, 1), device=values.device, dtype=values.dtype)
                valid_next_mask = th.zeros(bs - 1, device=values.device, dtype=th.bool)
            else:
                next_values = self.target_mac.evaluate_allocation(next_meta_batch)  # 使用target critic，更稳定
                if th.isnan(next_values).any():
                    print(f"Warning: NaN detected in next values at t_env={t_env}")
                    next_values = th.where(th.isnan(next_values), th.zeros_like(next_values), next_values)
                valid_next_mask = th.tensor([idx is not None for idx in next_indices],
                                           device=values.device, dtype=th.bool)
        
        # 计算TD targets和优势函数
        k = self.args.hier_agent['action_length']
        gamma_k = self.args.gamma ** k
        
        td_targets = th.zeros_like(values)  # (bs, 1)
        
        if bs > 1 and valid_next_mask.sum() > 0:
            idx = th.nonzero(valid_next_mask, as_tuple=False).squeeze(-1)
            valid_rewards = rewards[idx]
            valid_terminated = terminated[idx]
            assert next_values.shape[0] == idx.numel(), \
                f"next_values shape {next_values.shape[0]} != valid indices {idx.numel()}"
            td_targets_valid = (
                valid_rewards + 
                gamma_k * (1 - valid_terminated) * next_values
            ).detach()
            td_targets[idx] = td_targets_valid
        
        no_next_mask = th.zeros(bs, dtype=th.bool, device=values.device)
        if bs > 1:
            no_next_mask[:-1] = ~valid_next_mask
        no_next_mask[-1] = True
        
        mask_flat = mask.squeeze(-1)
        valid_no_next = no_next_mask & (mask_flat > 0.0)
        if valid_no_next.any():
            td_targets[valid_no_next] = rewards[valid_no_next].detach()
        
        # 准备mask
        policy_mask = mask  # (bs, 1)
        if self.args.hier_agent.get('decay_old', 0) > 0:
            cutoff = self.args.hier_agent['decay_old']
            ratio = (cutoff - t_env + meta_batch['t_added'].float()) / cutoff
            ratio = ratio.max(th.zeros_like(ratio))
            policy_mask = policy_mask * ratio
        
        # 关键修复：全batch归一化advantage（一次计算，所有minibatch共享）
        # 优势函数：A = TD_target - V(s)（TD residual）
        advantages = (td_targets - values).detach()  # (bs, 1)
        adv_mean = (advantages * policy_mask).sum() / (policy_mask.sum() + 1e-8)
        adv_var = (((advantages - adv_mean) ** 2) * policy_mask).sum() / (policy_mask.sum() + 1e-8)
        advantages_normalized = (advantages - adv_mean) / th.sqrt(adv_var + 1e-8)
        # 注意：这个advantages_normalized会在所有minibatch中复用，保持一致的标度
        
        # 2. 多个epoch更新（使用固定actions，不重新采样）
        # 准备minibatch indices（如果启用）
        if ppo_minibatch_size is not None and ppo_minibatch_size < bs:
            # 使用minibatch
            n_minibatches = (bs + ppo_minibatch_size - 1) // ppo_minibatch_size
        else:
            # 使用full batch
            ppo_minibatch_size = bs
            n_minibatches = 1
        
        for epoch in range(ppo_epochs):
            # 每个epoch打乱indices（minibatch shuffle）
            indices = th.randperm(bs, device=meta_batch['entities'].device)
            
            for minibatch_idx in range(n_minibatches):
                start_idx = minibatch_idx * ppo_minibatch_size
                end_idx = min((minibatch_idx + 1) * ppo_minibatch_size, bs)
                mb_indices = indices[start_idx:end_idx]
                
                # 提取minibatch数据（关键修复：使用白名单字段，更安全）
                required_keys = ['entities', 'entity_mask', 'entity2task_mask', 'task_mask', 
                               'obs_mask', 'last_alloc', 'avail_actions']
                mb_meta_batch = {}
                for key in required_keys:
                    if key in meta_batch:
                        v = meta_batch[key]
                        if isinstance(v, th.Tensor) and v.shape[0] == bs:
                            mb_meta_batch[key] = v[mb_indices]
                        else:
                            mb_meta_batch[key] = v
                
                mb_old_actions = old_actions[mb_indices]  # (mb_size, na, nt)
                mb_old_log_probs = old_log_probs[mb_indices]  # (mb_size, 1)
                mb_policy_mask = policy_mask[mb_indices].clone()  # (mb_size, 1) - 需要clone以便修改
                mb_td_targets = td_targets[mb_indices]  # (mb_size, 1)
                mb_advantages_normalized = advantages_normalized[mb_indices]  # (mb_size, 1) - 使用全batch归一化的advantage
                
                # 2.1 检查old_actions是否在当前avail_actions下可用（关键修复：过滤不可用动作）
                if 'avail_actions' in mb_meta_batch and mb_meta_batch['avail_actions'] is not None:
                    avail_actions_mb = mb_meta_batch['avail_actions']  # (mb_size, na, nt) 或 (mb_size, nt)
                    na = self.args.n_agents
                    if avail_actions_mb.dim() == 2:
                        avail_actions_mb = avail_actions_mb.unsqueeze(1).expand(-1, na, -1)  # (mb_size, na, nt)
                    
                    # 检查每个样本的old_actions是否可用
                    # old_actions是one-hot，需要检查对应位置是否在avail_actions中为1
                    action_indices = mb_old_actions.argmax(dim=-1)  # (mb_size, na) - 每个agent选择的task索引
                    mb_size = action_indices.shape[0]
                    valid_action_mask = th.ones(mb_size, device=mb_old_actions.device, dtype=th.bool)  # (mb_size,)
                    
                    for i in range(mb_size):
                        for ai in range(na):
                            task_idx = action_indices[i, ai].item()
                            if task_idx < avail_actions_mb.shape[2]:
                                if avail_actions_mb[i, ai, task_idx] < 0.5:  # 该任务不可用
                                    valid_action_mask[i] = False
                                    break
                    
                    # 将不可用动作样本的policy_mask置0
                    mb_policy_mask = mb_policy_mask * valid_action_mask.unsqueeze(-1).float()  # (mb_size, 1)
                    
                    # 记录不可用动作的比例（用于统计）
                    invalid_ratio = 1.0 - valid_action_mask.float().mean().item()
                    if epoch == 0 and minibatch_idx == 0 and invalid_ratio > 0:
                        if 'train_metrics/alloc_invalid_action_ratio' not in stats:
                            stats['train_metrics/alloc_invalid_action_ratio'] = invalid_ratio
                
                # 2.2 重新计算当前critic的values（用于critic更新，但advantage使用全batch归一化的）
                mb_values = self.mac.evaluate_allocation(mb_meta_batch)  # (mb_size, 1)
                
                # 2.3 使用当前策略计算固定actions的log_prob（关键修复：不重新采样）
                # 检查是否使用AR模式
                use_autoreg = self.args.hier_agent.get("use_autoreg", False) or \
                              (hasattr(self.mac.alloc_policy, 'compute_logprob_for_actions_autoreg') and 
                               self.args.hier_agent.get("pi_autoreg", False))
                
                if use_autoreg and hasattr(self.mac.alloc_policy, 'compute_logprob_for_actions_autoreg'):
                    # AR模式：使用自回归logprob计算
                    mb_alloc_order = None
                    if 'alloc_order' in meta_batch:
                        mb_alloc_order = meta_batch['alloc_order'][mb_indices]  # (mb_size, na)
                        # 注意：现在每个样本可能有不同的随机顺序，所以保持(mb_size, na)形状
                        # compute_logprob_for_actions_autoreg 会正确处理每个样本的顺序
                    new_log_probs, entropy = self.mac.alloc_policy.compute_logprob_for_actions_autoreg(
                        mb_meta_batch, mb_old_actions, mb_alloc_order
                    )
                else:
                    # 并行模式：使用原有的logprob计算
                    new_log_probs, entropy = self.mac.compute_allocation_logprob(mb_meta_batch, mb_old_actions)
                    # new_log_probs: (mb_size, 1), entropy: (mb_size, 1)
                
                # 检查-inf和NaN
                if th.isinf(new_log_probs).any() or th.isnan(new_log_probs).any():
                    n_inf = th.isinf(new_log_probs).sum().item()
                    n_nan = th.isnan(new_log_probs).sum().item()
                    if not hasattr(self, '_new_log_prob_inf_nan_count'):
                        self._new_log_prob_inf_nan_count = 0
                    self._new_log_prob_inf_nan_count += 1
                    if self._new_log_prob_inf_nan_count <= 1 or self._new_log_prob_inf_nan_count % 100 == 0:
                        print(f"Warning: inf/NaN in new_log_probs (inf: {n_inf}, NaN: {n_nan}, count: {self._new_log_prob_inf_nan_count}). Replacing with zeros.")
                    new_log_probs = th.where(
                        th.isinf(new_log_probs) | th.isnan(new_log_probs),
                        th.zeros_like(new_log_probs),
                        new_log_probs
                    )
                
                if th.isinf(mb_old_log_probs).any() or th.isnan(mb_old_log_probs).any():
                    n_inf = th.isinf(mb_old_log_probs).sum().item()
                    n_nan = th.isnan(mb_old_log_probs).sum().item()
                    if not hasattr(self, '_old_log_prob_inf_nan_count'):
                        self._old_log_prob_inf_nan_count = 0
                    self._old_log_prob_inf_nan_count += 1
                    if self._old_log_prob_inf_nan_count <= 1 or self._old_log_prob_inf_nan_count % 100 == 0:
                        print(f"Warning: inf/NaN in old_log_probs (inf: {n_inf}, NaN: {n_nan}, count: {self._old_log_prob_inf_nan_count}). Replacing with zeros.")
                    mb_old_log_probs = th.where(
                        th.isinf(mb_old_log_probs) | th.isnan(mb_old_log_probs),
                        th.zeros_like(mb_old_log_probs),
                        mb_old_log_probs
                    )
                
                # 2.4 计算重要性采样比率（关键修复：clamp log_ratio避免数值爆炸）
                log_ratio = (new_log_probs - mb_old_log_probs)  # (mb_size, 1)
                log_ratio = th.clamp(log_ratio, -20.0, 20.0)  # 防止exp爆炸
                ratio = th.exp(log_ratio)  # (mb_size, 1)
                
                # 2.5 PPO clipped objective（使用全batch归一化的advantages）
                surr1 = ratio * mb_advantages_normalized  # (mb_size, 1)
                surr2 = th.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * mb_advantages_normalized  # (mb_size, 1)
                policy_loss = -th.min(surr1, surr2) * mb_policy_mask  # (mb_size, 1)
                policy_loss = policy_loss.sum() / (mb_policy_mask.sum() + 1e-8)
                
                # 2.6 熵正则化（关键修复：使用mask）
                entropy_mean = (entropy * mb_policy_mask).sum() / (mb_policy_mask.sum() + 1e-8)
                entropy_loss = -entropy_mean
                
                pi_loss = (
                    policy_loss + 
                    self.args.hier_agent.get('entropy_loss', 0.01) * entropy_loss
                )
                
                # 2.7 更新Actor（包含蒸馏损失）
                self.alloc_pi_optimiser.zero_grad()
                
                # 精英蒸馏损失（如果启用且EliteBuffer非空）
                # 关键：蒸馏是辅助监督项，不参与PPO的ratio/advantage计算
                distill_loss = None
                elite_logprob_mean = None
                elite_lp_per_step = None
                
                use_distill = self.args.hier_agent.get('use_distill', False)
                if use_distill and self.elite_buffer is not None and len(self.elite_buffer) > 0:
                    distill_result = self._compute_distill_loss(t_env)
                    if distill_result is not None:
                        distill_loss, elite_logprob_mean = distill_result
                        if distill_loss is not None:
                            # 获取蒸馏权重（可以是固定值或动态调整）
                            distill_weight = self.args.hier_agent.get('distill_weight', 0.05)
                            # 合并损失：L_total = L_PPO + lambda_distill * L_distill
                            # 注意：这里只影响pi_loss，不影响PPO的ratio/advantage（它们已经在上面计算过了）
                            pi_loss = pi_loss + distill_weight * distill_loss
                            # 计算每步平均logprob（用于统计）
                            if elite_logprob_mean is not None:
                                na = self.args.n_agents
                                elite_lp_per_step = elite_logprob_mean / na
                
                pi_loss.backward()
                grad_norm = th.nn.utils.clip_grad_norm_(
                    self.alloc_pi_params, 
                    self.args.grad_norm_clip
                )
                self.alloc_pi_optimiser.step()
                
                # 2.8 更新Critic（关键修复：每个epoch/minibatch都更新，使用当前critic）
                # mb_values已经在上面计算过了，这里直接使用
                td_error = (mb_values - mb_td_targets.detach())  # (mb_size, 1)
                value_loss = (td_error ** 2 * mb_policy_mask).sum() / (mb_policy_mask.sum() + 1e-8)
                
                self.alloc_q_optimiser.zero_grad()
                value_loss.backward()
                grad_norm_v = th.nn.utils.clip_grad_norm_(
                    self.alloc_q_params, 
                    self.args.grad_norm_clip
                )
                self.alloc_q_optimiser.step()
                
                # 记录第一个epoch第一个minibatch的统计信息
                if epoch == 0 and minibatch_idx == 0:
                    stats['losses/alloc_policy_loss'] = policy_loss.cpu().item()
                    stats['losses/alloc_entropy'] = -entropy_loss.cpu().item()
                    stats['losses/alloc_total_loss'] = pi_loss.cpu().item()
                    stats['losses/alloc_value_loss'] = value_loss.cpu().item()
                    stats['train_metrics/alloc_pi_grad_norm'] = grad_norm
                    stats['train_metrics/alloc_value_grad_norm'] = grad_norm_v
                    stats['train_metrics/alloc_ratio_mean'] = ratio.mean().cpu().item()
                    stats['train_metrics/alloc_ratio_std'] = ratio.std().cpu().item()
                    stats['train_metrics/alloc_ratio_min'] = ratio.min().cpu().item()
                    stats['train_metrics/alloc_ratio_max'] = ratio.max().cpu().item()
                    stats['train_metrics/alloc_log_ratio_mean'] = log_ratio.mean().cpu().item()
                    # 记录蒸馏相关统计
                    if distill_loss is not None:
                        stats['losses/alloc_distill_loss'] = distill_loss.item()
                        distill_weight = self.args.hier_agent.get('distill_weight', 0.05)
                        stats['train_metrics/alloc_distill_weight'] = distill_weight
                        stats['train_metrics/elite_buffer_size'] = len(self.elite_buffer)
                    if elite_logprob_mean is not None:
                        stats['train_metrics/elite_logprob_mean'] = elite_logprob_mean
                    if elite_lp_per_step is not None:
                        stats['train_metrics/elite_lp_per_step'] = elite_lp_per_step
                    stats['train_metrics/alloc_log_ratio_std'] = log_ratio.std().cpu().item()
                    stats['train_metrics/alloc_log_ratio_min'] = log_ratio.min().cpu().item()
                    stats['train_metrics/alloc_log_ratio_max'] = log_ratio.max().cpu().item()
                    # 检查是否有-inf
                    stats['train_metrics/alloc_new_log_prob_inf_count'] = float(th.isinf(new_log_probs).sum().cpu().item())
                    stats['train_metrics/alloc_old_log_prob_inf_count'] = float(th.isinf(mb_old_log_probs).sum().cpu().item())
                    stats['train_metrics/alloc_entropy_mean'] = entropy_mean.cpu().item()
        
        # 3. 更新目标网络
        if (episode_num - self.last_alloc_target_update_episode) / \
           self.args.alloc_target_update_interval >= 1.0:
            self._update_alloc_targets()
            self.last_alloc_target_update_episode = episode_num
        
        # 4. 记录统计信息
        # 确保这些基础统计信息总是被记录（不依赖于epoch/minibatch条件）
        stats['train_metrics/alloc_batch_size'] = float(policy_mask.sum().cpu().item())
        stats['train_metrics/alloc_valid_samples'] = float((policy_mask > 0).sum().cpu().item())
        stats['train_metrics/alloc_advantage_mean'] = advantages.mean().cpu().item()
        stats['train_metrics/alloc_advantage_std'] = advantages.std().cpu().item()
        
        # 计算分配相关的metrics（与A2C保持一致）
        # 使用old_actions作为当前分配（因为这是实际执行的分配）
        current_alloc = old_actions  # (bs, na, nt)
        
        # 计算任务分布（每个任务被分配的智能体数量）
        active_ag = 1 - meta_batch['entity_mask'][:, :self.args.n_agents].float()  # (bs, na)
        ag_per_task = (current_alloc * active_ag.unsqueeze(-1)).sum(dim=1)  # (bs, nt) - 每个任务被分配的智能体数量
        total_active_ag = active_ag.sum(dim=1, keepdim=True)  # (bs, 1)
        
        # 归一化得到任务分布（每个任务占用的智能体比例）
        task_distribution = ag_per_task / (total_active_ag + 1e-8)  # (bs, nt)
        
        # 计算任务分布熵（衡量分配的均匀性）
        task_distribution_entropy = -(task_distribution * (task_distribution + 1e-8).log()).sum(dim=1)  # (bs,)
        stats['alloc_metrics/task_distribution_entropy'] = task_distribution_entropy.mean().cpu().item()
        stats['alloc_metrics/task_distribution_max'] = task_distribution.max().cpu().item()
        stats['alloc_metrics/task_distribution_min'] = task_distribution.min().cpu().item()
        
        # 计算智能体任务分配变化百分比（如果有last_alloc）
        if 'last_alloc' in meta_batch and meta_batch['last_alloc'] is not None:
            last_alloc = meta_batch['last_alloc']  # (bs, na, nt)
            ag_changed = (last_alloc.argmax(dim=2) != current_alloc.argmax(dim=2)).float()  # (bs, na)
            prev_al_exists = (last_alloc.sum(dim=(1, 2)) >= 1).float()  # (bs,)
            perc_changed_per_step = ((ag_changed * active_ag).sum(dim=1) / (active_ag.sum(dim=1) + 1e-8))  # (bs,)
            if prev_al_exists.sum() > 0:
                perc_changed = (perc_changed_per_step * prev_al_exists).sum() / prev_al_exists.sum()
                stats['alloc_metrics/perc_ag_changed'] = perc_changed.cpu().item()
            else:
                stats['alloc_metrics/perc_ag_changed'] = 0.0
        
        # 计算智能体-任务集中度（智能体数量与任务实体数量的差异）
        nonagent2task = 1 - meta_batch['entity2task_mask'][:, self.args.n_agents:].float()  # (bs, n_nonagent, nt)
        nag_per_task = nonagent2task.sum(dim=1)  # (bs, nt) - 每个任务的非智能体实体数量
        absdiff_per_task = (ag_per_task - nag_per_task).abs()  # (bs, nt)
        active_task_mask = 1 - meta_batch['task_mask'].float()  # (bs, nt)
        abs_diff_mean = absdiff_per_task.sum(dim=1) / (active_task_mask.sum(dim=1) + 1e-8)  # (bs,)
        stats['alloc_metrics/ag_task_concentration'] = abs_diff_mean.mean().cpu().item()
        
        # 检查是否有任务没有分配到智能体
        active_task = 1 - meta_batch['task_mask'].float()  # (bs, nt)
        task_has_agents = (ag_per_task > 0).float()  # (bs, nt)
        any_task_no_agents = ((task_has_agents.sum(dim=1, keepdim=True) != active_task.sum(dim=1, keepdim=True))).float()  # (bs, 1)
        stats['alloc_metrics/any_task_no_agents'] = any_task_no_agents.mean().cpu().item()
        
        # 记录统计信息（如果满足时间间隔条件）
        if t_env - self.log_alloc_stats_t >= self.args.learner_log_interval:
            # 确保stats不为空才记录
            if stats:
                for name, value in stats.items():
                    self.logger.log_stat(name, value, t_env)
            self.log_alloc_stats_t = t_env
        
        return stats, old_actions  # 返回固定的actions，不是新采样的

    def _broadcast_decisions_to_batch(self, decisions, decision_pts):
        decision_pts = decision_pts.squeeze(-1)
        bs, ts = decision_pts.shape
        bcast_decisions = {k: th.zeros_like(v[[0]]).unsqueeze(0).repeat(bs * rep, ts, *(1 for _ in range(len(v.shape) - 1))) for k, (v, rep) in decisions.items()}
        for decname in bcast_decisions:
            value, rep = decisions[decname]
            bcast_decisions[decname][decision_pts.repeat(rep, 1)] = value
        for t in range(1, ts):
            for decname in bcast_decisions:
                rep = decisions[decname][1]
                prev_value = bcast_decisions[decname][:, t - 1]
                bcast_decisions[decname][:, t] = ((decision_pts[:, t].repeat(rep).reshape(bs * rep, 1, 1).to(prev_value.dtype) * bcast_decisions[decname][:, t])
                                                  + ((1 - decision_pts[:, t].repeat(rep).reshape(bs * rep, 1, 1)).to(prev_value.dtype) * prev_value))
        return bcast_decisions

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        # episode over (not including timeout) - determines when to bootstrap
        terminated = batch["terminated"][:, :-1].float()
        # env reset (either terminated or timed out) - determines what timesteps
        # to learn from - we can't learn from final ts bc there is no
        # transition
        reset = batch["reset"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - reset[:, :-1])
        org_mask = mask.clone()
        avail_actions = batch["avail_actions"]
        if self.args.agent['subtask_cond'] is not None:
            # Learning separate controllers for each task
            rewards = batch['task_rewards'][:, :-1]
            terminated = batch['tasks_terminated'][:, :-1].float()
            mask = mask.repeat(1, 1, self.args.n_tasks)
            mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
            task_has_agents = (1 - batch['entity2task_mask'][:, :-1, :self.args.n_agents]).sum(2) > 0
            mask *= task_has_agents.float()

        # # Calculate estimated Q-Values
        # mac_out = []
        self.mac.init_hidden(batch.batch_size)
        # enable things like dropout on mac and mixer, but not target_mac and target_mixer
        self.mac.train()
        self.target_mac.eval()
        if self.mixer is not None:
            self.mixer.train()
            self.target_mixer.eval()

        coach_h = None
        targ_coach_h = None
        coach_z = None
        targ_coach_z = None

        imagine_inps = None
        if self.args.agent['imagine']:
            imagine_inps, imagine_groups = self.mac.agent.make_imagined_inputs(batch)
        if self.use_copa:
            coach_h = self.mac.coach.encode(batch, imagine_inps=imagine_inps)
            targ_coach_h = self.target_mac.coach.encode(batch)
            decision_points = batch['hier_decision'].squeeze(-1)
            bs_rep = 1
            if self.args.agent['imagine']:
                bs_rep = 3
            coach_h_t0 = coach_h[decision_points.repeat(bs_rep, 1)]
            targ_coach_h_t0 = targ_coach_h[decision_points]
            coach_z_t0, coach_mu_t0, coach_logvar_t0 = self.mac.coach.strategy(coach_h_t0)
            coach_mu_t0 = coach_mu_t0.chunk(bs_rep, dim=0)[0]
            coach_logvar_t0 = coach_logvar_t0.chunk(bs_rep, dim=0)[0]
            targ_coach_z_t0, _, _ = self.target_mac.coach.strategy(targ_coach_h_t0)

            bcast_ins = {
                'coach_z_t0': (coach_z_t0, bs_rep),
                'coach_mu_t0': (coach_mu_t0, 1),
                'coach_logvar_t0': (coach_logvar_t0, 1),
                'targ_coach_z_t0': (targ_coach_z_t0, 1),
            }
            bcast_decisions = self._broadcast_decisions_to_batch(bcast_ins, batch['hier_decision'])
            coach_z = bcast_decisions['coach_z_t0']
            coach_mu = bcast_decisions['coach_mu_t0']
            coach_logvar = bcast_decisions['coach_logvar_t0']
            targ_coach_z = bcast_decisions['targ_coach_z_t0']


        batch_mult = 1
        if self.args.agent['imagine']:
            batch_mult += 2

        all_mac_out, mac_info = self.mac.forward(
            batch, t=None,
            coach_z=coach_z,
            imagine_inps=imagine_inps)
        rep_actions = actions.repeat(batch_mult, 1, 1, 1)
        all_chosen_action_qvals = th.gather(all_mac_out[:, :-1], dim=3, index=rep_actions).squeeze(3)  # Remove the last dim

        mac_out_tup = all_mac_out.chunk(batch_mult, dim=0)
        caq_tup = all_chosen_action_qvals.chunk(batch_mult, dim=0)

        mac_out = mac_out_tup[0]
        chosen_action_qvals = caq_tup[0]
        if self.args.agent['imagine']:
            caq_imagine = th.cat(caq_tup[1:], dim=2)

        self.target_mac.init_hidden(batch.batch_size)

        target_mac_out, _ = self.target_mac.forward(batch, coach_z=targ_coach_z, t=None, target=True)
        if self.args.agent['subtask_cond'] is not None:
            allocs = (1 - batch['entity2task_mask'][:, :, :self.args.n_agents])
            avail_actions_targ = parse_avail_actions(avail_actions[:, 1:], allocs[:, :-1], self.args)
        else:
            avail_actions_targ = avail_actions[:, 1:]
        target_mac_out = target_mac_out[:, 1:]

        # Mask out unavailable actions
        target_mac_out[avail_actions_targ == 0] = -9999999  # From OG deepmarl

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()[:, 1:]
            mac_out_detach[avail_actions_targ == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            mix_ins, targ_mix_ins = self._get_mixer_ins(batch)

            chosen_action_qvals = self.mixer(chosen_action_qvals, mix_ins)
            gamma = self.args.gamma

            target_max_qvals = self.target_mixer(target_max_qvals, targ_mix_ins)
            target_max_qvals = self.target_mixer.denormalize(target_max_qvals)
            # Calculate 1-step Q-Learning targets
            targets = (rewards + gamma * (1 - terminated) * target_max_qvals).detach()
            if self.args.popart:
                targets = self.mixer.popart_update(
                    targets, mask)

            if self.args.agent['imagine']:
                # don't need last timestep
                imagine_groups = [gr[:, :-1] for gr in imagine_groups]
                caq_imagine = self.mixer(caq_imagine, mix_ins,
                                         imagine_groups=imagine_groups)
        else:
            targets = (rewards + self.args.gamma * (1 - terminated) * target_max_qvals).detach()

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        if self.args.agent['imagine']:
            im_prop = self.args.lmbda
            im_td_error = (caq_imagine - targets.detach())
            im_masked_td_error = im_td_error * mask
            im_loss = (im_masked_td_error ** 2).sum() / mask.sum()
            loss = (1 - im_prop) * loss + im_prop * im_loss

        if self.use_copa and self.args.hier_agent['copa_vi_loss']:
            # VI loss
            q_mu, q_logvar = self.mac.copa_recog(batch)
            q_t = D.normal.Normal(q_mu, (0.5 * q_logvar).exp())
            coach_z = coach_z.chunk(bs_rep, dim=0)[0]  # if combining with REFIL, only train full info Z
            log_prob = q_t.log_prob(coach_z).clamp_(-1000, 0).sum(-1)
            # entropy loss
            p_ = D.normal.Normal(coach_mu, (0.5 * coach_logvar).exp())
            entropy = p_.entropy().clamp_(0, 10).sum(-1)

            # mask inactive agents
            agent_mask = 1 - batch['entity_mask'][:, :, :self.args.n_agents].float()
            log_prob = (log_prob * agent_mask).sum(-1) / (agent_mask.sum(-1) + 1e-8)
            entropy = (entropy * agent_mask).sum(-1) / (agent_mask.sum(-1) + 1e-8)

            vi_loss = (-log_prob[:, :-1] * org_mask.squeeze(-1)).sum() / org_mask.sum()
            entropy_loss = (-entropy[:, :-1] * org_mask.squeeze(-1)).sum() / org_mask.sum()
            
            loss += vi_loss * self.args.vi_lambda + entropy_loss * self.args.vi_lambda / 10

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("losses/q_loss", loss.item(), t_env)
            if self.args.agent['imagine']:
                self.logger.log_stat("losses/im_loss", im_loss.item(), t_env)
            if self.use_copa and self.args.hier_agent['copa_vi_loss']:
                self.logger.log_stat("losses/copa_vi_loss", vi_loss.item(), t_env)
                self.logger.log_stat("losses/copa_entropy_loss", entropy_loss.item(), t_env)
            self.logger.log_stat("train_metrics/q_grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            # 修复：避免除零错误
            if mask_elems > 0:
                self.logger.log_stat("train_metrics/td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
                self.logger.log_stat("train_metrics/q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
                self.logger.log_stat("train_metrics/target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            else:
                # 如果mask全为0，记录默认值
                self.logger.log_stat("train_metrics/td_error_abs", 0.0, t_env)
                self.logger.log_stat("train_metrics/q_taken_mean", 0.0, t_env)
                self.logger.log_stat("train_metrics/target_mean", 0.0, t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def _update_alloc_targets(self):
        self.target_mac.load_alloc_state(self.mac)
        self.logger.console_logger.info("Updated allocation target network")
    
    def _compute_distill_loss(self, t_env: int):
        """
        计算精英蒸馏损失（带门控机制，防止拉崩策略）
        
        数学定义：
        L_distill(θ) = -E_{(s,o,a)~E} [log π_θ(a|s,o)]
        
        本质上是KL散度的简化形式：
        KL(π_teacher || π_student) = E_{a~π_teacher} [log π_teacher(a) - log π_student(a)]
        由于π_teacher是one-hot分布（EA精英），log π_teacher(a) = 0，因此简化为：
        L_distill(θ) = -E_{a~π_teacher} [log π_student(a)]
        
        关键点：
        - 精英数据不参与PPO的ratio/advantage，只作为辅助监督项
        - 加入logprob门控，防止蒸馏将策略拉向当前分布完全覆盖不到的区域
        - 早期禁用：elite buffer size < 20时不蒸馏，避免质量不稳定的精英拉偏策略
        
        Args:
            t_env: 当前环境时间步
        
        Returns:
            (distill_loss, elite_logprob_mean) 或 None
            - distill_loss: 蒸馏损失（如果logprob过低或早期禁用返回None）
            - elite_logprob_mean: 精英logprob均值（用于监控）
        """
        if self.elite_buffer is None or len(self.elite_buffer) == 0:
            return None
        
        # 早期禁用条件：训练早期精英质量不稳定，蒸馏会把PPO拉向某些偶然高分但结构错误的轨迹
        distill_warmup_elites = self.args.hier_agent.get('distill_warmup_elites', 20)
        if len(self.elite_buffer) < distill_warmup_elites:
            # elite buffer size < 20时不蒸馏，等待积累足够多的高质量精英
            return None
        
        # 从EliteBuffer采样精英数据
        distill_batch_size = self.args.hier_agent.get('distill_batch_size', 16)
        elite_batch = self.elite_buffer.sample(batch_size=distill_batch_size)
        
        if elite_batch is None:
            return None
        
        elite_meta = elite_batch["meta_inputs"]
        elite_actions = elite_batch["alloc_actions"]      # (bs_e, na, nt)
        elite_order = elite_batch["alloc_order"]          # (bs_e, na)
        
        if elite_actions is None or elite_order is None:
            return None
        
        # 计算当前策略对精英动作序列的logprob（行为克隆）
        try:
            elite_logprob, _ = self.mac.alloc_policy.compute_logprob_for_actions_autoreg(
                elite_meta, elite_actions, elite_order
            )
        except Exception as e:
            # 如果计算失败（可能是维度问题），跳过这次蒸馏
            print(f"Warning: Failed to compute distill loss: {e}. Skipping.")
            return None
        
        # 确保elite_logprob是标量或1维tensor
        if elite_logprob.dim() > 1:
            elite_logprob = elite_logprob.view(-1)
        elite_logprob_mean = elite_logprob.mean()
        
        # 蒸馏门控：使用"每步平均logprob"而不是总logprob，更合理且可泛化
        # elite_logprob是整条分配序列的logprob（na步相加），归一化到每步平均
        na = elite_order.shape[1]  # 智能体数量
        elite_lp_per_step = elite_logprob_mean / na  # 每步平均logprob
        
        # 阈值有明确语义："每一步平均概率不能低到离谱"
        min_lp_per_step = self.args.hier_agent.get('distill_min_lp_per_step', -2.0)
        if elite_lp_per_step.item() < min_lp_per_step:
            # 每步平均logprob太低，说明当前策略完全覆盖不到这个精英分配，跳过蒸馏
            return (None, elite_logprob_mean.item())
        
        # 蒸馏损失：L_distill = -mean(log_prob)
        # 含义：提高PPO在这些"成功状态"下生成这些分配的概率
        # 本质上是KL(π_teacher || π_student)的简化形式
        distill_loss = -elite_logprob_mean
        
        return (distill_loss, elite_logprob_mean.item())
    
    def _get_distill_weight(self, t_env: int) -> float:
        """
        获取蒸馏损失权重（动态调整）
        
        训练前期：0.01
        训练中期：0.05
        训练后期（冲上限）：0.1
        
        Args:
            t_env: 当前环境时间步
        
        Returns:
            lambda_distill权重
        """
        # 获取训练进度（0.0到1.0）
        progress = min(1.0, t_env / self.args.t_max) if hasattr(self.args, 't_max') else 0.0
        
        # 线性插值
        if progress < 0.33:
            # 训练前期
            lambda_distill = 0.01
        elif progress < 0.67:
            # 训练中期
            lambda_distill = 0.01 + (0.05 - 0.01) * (progress - 0.33) / 0.34
        else:
            # 训练后期（冲上限）
            lambda_distill = 0.05 + (0.1 - 0.05) * (progress - 0.67) / 0.33
        
        return lambda_distill

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}opt.th".format(path))

    def load_models(self, path, pi_only=False, evaluate=False):
        self.mac.load_models(path, pi_only=pi_only)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path, pi_only=pi_only)
        if not evaluate and not pi_only:
            if self.mixer is not None:
                self.mixer.load_state_dict(th.load("{}mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.optimiser.load_state_dict(th.load("{}opt.th".format(path), map_location=lambda storage, loc: storage))
