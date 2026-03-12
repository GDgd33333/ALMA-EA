from .basic_controller import BasicMAC
import torch as th
from modules.agents import ALLOC_CRITIC_REGISTRY, ALLOC_POLICY_REGISTRY
from modules.agents.copa import Coach, RecognitionModel
from components.epsilon_schedules import DecayThenFlatSchedule
from utils.allocation import random_allocs


# This multi-agent controller shares parameters between agents and takes
# entities + observation masks as input
class EntityMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(EntityMAC, self).__init__(scheme, groups, args)
        self.task_allocations = None
        e_start_val, e_finish_val, e_decay_time_frac = args.hier_agent['alloc_eps'].split('-')
        e_start_val = float(e_start_val)
        e_finish_val = float(e_finish_val)
        e_decay_time_frac = float(e_decay_time_frac)
        e_decay_time = int(e_decay_time_frac * self.args.t_max)
        
        # 修复：如果decay_time为0，避免除零错误
        # 对于PPO（不需要epsilon-greedy），如果start和finish都是0，设置一个最小的decay_time
        if e_decay_time == 0:
            if e_start_val == 0.0 and e_finish_val == 0.0:
                # PPO模式：不需要epsilon-greedy，创建一个始终返回0的schedule
                # 设置decay_time=1避免除零，但schedule会立即返回finish值（0）
                e_decay_time = 1
            else:
                # 如果start或finish不为0但decay_time为0，设置最小值为1
                e_decay_time = 1
        
        self.alloc_eps_schedule = DecayThenFlatSchedule(e_start_val, e_finish_val, e_decay_time, decay="linear")
        self.alloc_eps = e_start_val

        self.prop_alloc_eps = 0
        self.prop_alloc_eps_schedule = None
        if args.hier_agent['prop_alloc_eps'] != '':
            ce_start_val, ce_finish_val, ce_decay_time_frac = args.hier_agent['prop_alloc_eps'].split('-')
            ce_decay_time = int(float(ce_decay_time_frac) * self.args.t_max)
            self.prop_alloc_eps_schedule = DecayThenFlatSchedule(float(ce_start_val), float(ce_finish_val), ce_decay_time, decay="linear")
            self.prop_alloc_eps = float(ce_start_val)

    def alloc_pi_params(self):
        return self.alloc_policy.parameters()

    def alloc_q_params(self):
        return self.alloc_critic.parameters()

    def compute_allocation(self, meta_batch, t_env=None, t_ep=None, acting=False,
                           test_mode=False, calc_stats=False, target_mac=None,
                           ep_batch=None, **kwargs):
        # 检查是否使用A2C或PPO（不再生成多个提案）
        is_a2c_or_ppo = self.args.hier_agent["task_allocation"] in ["a2c", "ppo"]
        # 检查是否使用自回归AR模式
        use_autoreg = self.args.hier_agent.get("use_autoreg", False) or \
                      (hasattr(self.alloc_policy, 'compute_allocation_autoreg') and 
                       self.args.hier_agent.get("pi_autoreg", False))
        
        if is_a2c_or_ppo:
            # A2C/PPO模式：直接采样单个动作
            # 对于PPO，在rollout时需要保存旧策略的log_prob
            # 关键修复：在EA评估时（test_mode=True），也需要保存old_actions用于精英蒸馏
            # 检查是否是EA评估：通过检查是否有特殊的标志（通过kwargs传入）
            is_ea_evaluation = kwargs.get('is_ea_evaluation', False)
            need_old_log_prob = (acting and 
                               self.args.hier_agent["task_allocation"] == "ppo" and
                               ep_batch is not None and
                               (not test_mode or is_ea_evaluation))  # EA评估时即使test_mode=True也保存
            
            # 根据是否使用AR模式选择不同的采样方法
            if use_autoreg and hasattr(self.alloc_policy, 'compute_allocation_autoreg'):
                # AR模式：使用自回归采样
                allocs, log_prob, entropy, alloc_order = self.alloc_policy.compute_allocation_autoreg(
                    meta_batch, test_mode=test_mode
                )
                if calc_stats or need_old_log_prob:
                    stats = {
                        'log_prob': log_prob,
                        'entropy': entropy
                    }
                else:
                    stats = None
            else:
                # 并行模式：使用原有的forward方法
                result = self.alloc_policy(meta_batch, calc_stats=calc_stats or need_old_log_prob, test_mode=test_mode, **kwargs)
                if calc_stats or need_old_log_prob:
                    allocs, stats = result
                else:
                    allocs = result
                    stats = None
                alloc_order = None  # 并行模式不需要order
            
            # PPO: 在rollout时保存旧策略的log_prob和actions到episode_batch
            # 注意：需要在epsilon-greedy之后保存，确保保存的是实际执行的动作
            # 但这个函数在epsilon-greedy之前调用，所以我们需要在epsilon-greedy之后再次保存
            # 暂时先保存采样得到的actions，后续在epsilon-greedy之后会更新
            if need_old_log_prob and stats is not None and 'log_prob' in stats and t_ep is not None:
                old_log_prob = stats['log_prob'].detach()  # (n_dp, 1) - n_dp是决策点数量
                old_actions = allocs.detach()  # (n_dp, na, nt) - n_dp是决策点数量
                # 找到决策点的位置
                if isinstance(t_ep, int):
                    decision_pts = ep_batch['hier_decision'][:, t_ep].flatten()
                    t_ep_idx = t_ep
                else:
                    # t_ep是slice，取第一个元素
                    t_ep_idx = t_ep.start if hasattr(t_ep, 'start') and t_ep.start is not None else 0
                    decision_pts = ep_batch['hier_decision'][:, t_ep_idx].flatten()
                d_inds = (decision_pts == 1)
                n_dp = d_inds.sum().item()  # 决策点数量
                
                if n_dp > 0:
                    # 关键修复：old_log_prob的batch size是n_dp（决策点数量），不是ep_batch的batch size
                    # 需要确保old_log_prob和old_actions的batch size与决策点数量匹配
                    if old_log_prob.shape[0] != n_dp:
                        # 如果形状不匹配，说明meta_batch可能包含了非决策点的数据
                        # 这种情况下，我们需要从old_log_prob中提取对应决策点的数据
                        # 但通常meta_batch只包含决策点，所以old_log_prob的batch size应该等于n_dp
                        print(f"Warning: old_log_prob batch size {old_log_prob.shape[0]} != n_dp {n_dp}")
                        # 如果old_log_prob的batch size大于n_dp，取前n_dp个
                        if old_log_prob.shape[0] > n_dp:
                            old_log_prob = old_log_prob[:n_dp]
                            old_actions = old_actions[:n_dp]
                        # 如果old_log_prob的batch size小于n_dp，说明有问题，跳过
                        elif old_log_prob.shape[0] < n_dp:
                            print(f"Error: old_log_prob batch size {old_log_prob.shape[0]} < n_dp {n_dp}, skipping")
                            n_dp = 0
                    
                    if n_dp > 0:
                        # 将old_log_prob和old_actions保存到episode_batch
                        # 使用d_inds的非零索引来定位决策点
                        dp_indices = th.nonzero(d_inds, as_tuple=False).squeeze(-1)  # (n_dp,)
                        
                        if 'old_log_prob' in ep_batch.data.transition_data:
                            # 只保存决策点的log_prob
                            # 使用dp_indices来索引ep_batch，用range(n_dp)来索引old_log_prob
                            # 保持old_log_prob的(n_dp, 1)形状，与索引结果(n_dp, 1)匹配
                            ep_batch.data.transition_data['old_log_prob'][dp_indices, t_ep_idx] = old_log_prob  # (n_dp, 1)
                        
                        if 'old_actions' in ep_batch.data.transition_data:
                            # 只保存决策点的actions
                            ep_batch.data.transition_data['old_actions'][dp_indices, t_ep_idx] = old_actions  # (n_dp, na, nt)
                        
                        # AR模式：保存alloc_order（如果存在）
                        if use_autoreg and alloc_order is not None:
                            if 'alloc_order' in ep_batch.data.transition_data:
                                # alloc_order: (na,) 或 (n_dp, na) 或 (bs, na) 其中 bs >= n_dp
                                if alloc_order.dim() == 1:
                                    # 如果是1维，扩展到batch维度（固定顺序的情况）
                                    alloc_order_expanded = alloc_order.unsqueeze(0).repeat(n_dp, 1)  # (n_dp, na)
                                elif alloc_order.shape[0] >= n_dp:
                                    # 如果是2维且batch size >= n_dp，取前n_dp个（随机顺序的情况）
                                    alloc_order_expanded = alloc_order[:n_dp]  # (n_dp, na)
                                else:
                                    # 如果batch size < n_dp，扩展到n_dp（不应该发生，但做保护）
                                    alloc_order_expanded = alloc_order.repeat((n_dp // alloc_order.shape[0] + 1), 1)[:n_dp]  # (n_dp, na)
                                ep_batch.data.transition_data['alloc_order'][dp_indices, t_ep_idx] = alloc_order_expanded  # (n_dp, na)
            
            # 关键修复：无论是否need_old_log_prob，只要产生了allocs，就写入alloc_actions字段
            # 这样在EA评估时（test_mode=True）也能获取到高层分配动作
            if acting and ep_batch is not None and t_ep is not None:
                # 找到决策点的位置
                if isinstance(t_ep, int):
                    decision_pts = ep_batch['hier_decision'][:, t_ep].flatten()
                    t_ep_idx = t_ep
                else:
                    t_ep_idx = t_ep.start if hasattr(t_ep, 'start') and t_ep.start is not None else 0
                    decision_pts = ep_batch['hier_decision'][:, t_ep_idx].flatten()
                d_inds = (decision_pts == 1)
                n_dp = d_inds.sum().item()  # 决策点数量
                
                if n_dp > 0:
                    dp_indices = th.nonzero(d_inds, as_tuple=False).squeeze(-1)  # (n_dp,)
                    
                    # 保存alloc_actions到专门的字段（用于精英蒸馏）
                    if 'alloc_actions' in ep_batch.data.transition_data:
                        # allocs的shape是(n_dp, na, nt)，需要匹配
                        if allocs.shape[0] >= n_dp:
                            allocs_to_save = allocs[:n_dp].detach()  # (n_dp, na, nt)
                            ep_batch.data.transition_data['alloc_actions'][dp_indices, t_ep_idx] = allocs_to_save
        else:
            # AQL模式：生成多个提案，选择最优
            all_allocs = self.alloc_policy(meta_batch, calc_stats=calc_stats, n_proposals=self.args.hier_agent['n_proposals'], test_mode=test_mode, **kwargs)
            if calc_stats:
                all_allocs, stats = all_allocs
                stats['all_allocs'] = all_allocs
            # evaluate allocations and take best
            evaluations = self.evaluate_allocation(meta_batch, override_alloc=all_allocs, test_mode=test_mode)
            evaluations = evaluations.squeeze(2)
            best_prop_inds = evaluations.argmax(dim=1)
            allocs = all_allocs[th.arange(all_allocs.shape[0]), best_prop_inds]
            if calc_stats:
                stats['best_prop_inds'] = best_prop_inds  # used to select which action to maximize log_prob of
                if target_mac is not None:
                    stats['targ_best_prop_values'] = target_mac.evaluate_allocation(
                        meta_batch, override_alloc=allocs, test_mode=test_mode)

        if acting and not test_mode:
            # epsilon greedy（PPO不使用epsilon-greedy，主要靠采样+entropy探索）
            # A2C可以使用epsilon-greedy，但PPO建议不使用
            use_eps_greedy = self.args.hier_agent["task_allocation"] != "ppo"
            
            if use_eps_greedy:
                rand_allocs = random_allocs(meta_batch['task_mask'], meta_batch['entity_mask'], self.n_agents)
                assert t_env is not None, "Must provide t_env for epsilon greedy exploration schedule"
                assert t_ep is not None, "Must provide t_ep for epsilon greedy exploration schedule"
                self.alloc_eps = self.alloc_eps_schedule.eval(t_env)
                
                # prop_alloc_eps只在AQL模式下使用（A2C/PPO不需要）
                if self.args.hier_agent["task_allocation"] == "aql" and self.prop_alloc_eps_schedule is not None:
                    self.prop_alloc_eps = self.prop_alloc_eps_schedule.eval(t_env)
                    # We do pure random sampling independently per agent, while we
                    # copy entire proposed allocations (since they're sampled
                    # autoregressively and agents' assignments depend on other
                    # agents). Pure random will overwrite proposals, as it is
                    # applied second.
                    prop_draw = th.rand_like(rand_allocs[:, 0, 0])
                    prop_eps_mask = (prop_draw <= self.prop_alloc_eps)
                    allocs[prop_eps_mask] = all_allocs[:, 0][prop_eps_mask]  # take randomly sampled proposal action

                # epsilon greedy（A2C和AQL使用）
                eps_draw = th.rand_like(rand_allocs[:, :, 0])
                eps_mask = (eps_draw <= self.alloc_eps)
                allocs[eps_mask] = rand_allocs[eps_mask]
        if calc_stats:
            return allocs, stats
        return allocs, None  # 始终返回两个值以保持接口一致性


    def evaluate_allocation(self, meta_batch, **kwargs):
        return self.alloc_critic(meta_batch, **kwargs)
    
    def compute_allocation_logprob(self, meta_batch, actions_fixed):
        """
        计算固定actions的log_prob（用于PPO训练）
        
        Args:
            meta_batch: 包含状态信息的字典
            actions_fixed: (bs, na, nt) - 固定的actions（one-hot编码）
        
        Returns:
            log_prob: (bs, 1) - 联合log概率
            entropy: (bs, 1) - 联合熵
        """
        if self.args.hier_agent["task_allocation"] in ["a2c", "ppo"]:
            return self.alloc_policy.compute_logprob_for_actions(meta_batch, actions_fixed)
        else:
            raise NotImplementedError("compute_allocation_logprob only supported for A2C/PPO")

    def _make_meta_batch(self, ep_batch, t_ep):
        # Add quantities necessary for meta-controller (only used for acting as
        # this doesn't compute rewards/term, etc.)
        decision_pts = ep_batch['hier_decision'][:, t_ep].flatten()
        d_inds = (decision_pts == 1)
        meta_batch = {
            'entities': ep_batch['entities'][d_inds, t_ep],
            'obs_mask': ep_batch['obs_mask'][d_inds, t_ep],
            'entity_mask': ep_batch['entity_mask'][d_inds, t_ep],
            'entity2task_mask': ep_batch['entity2task_mask'][d_inds, t_ep],
            'task_mask': ep_batch['task_mask'][d_inds, t_ep],
            'avail_actions': ep_batch['avail_actions'][d_inds, t_ep],
        }
        if self.learned_alloc:
            meta_batch['last_alloc'] = self.task_allocations[d_inds]
        return meta_batch

    def _build_agents(self, input_shapes):
        agent_input_shape, hier_input_shape = input_shapes
        super()._build_agents(agent_input_shape)
        if self.learned_alloc:
            self.alloc_critic = ALLOC_CRITIC_REGISTRY[self.args.hier_agent['alloc_critic']](hier_input_shape, self.args)
            self.alloc_policy = ALLOC_POLICY_REGISTRY[self.args.hier_agent['alloc_policy']](hier_input_shape, self.args)
        if self.use_copa:
            self.coach = Coach(self.args)
            if self.args.hier_agent['copa_vi_loss']:
                self.copa_recog = RecognitionModel(self.args)


    def _build_inputs(self, batch, t, target=False, imagine_inps=None):
        # Assumes homogenous agents with entity + observation mask inputs.
        bs = batch.batch_size
        entity_parts = []
        entity_parts.append(batch["entities"][:, t])  # bs, ts, n_entities, vshape
        if self.args.entity_last_action:
            ent_acs = th.zeros(bs, t.stop - t.start, self.args.n_entities,
                               self.args.n_actions, device=batch.device,
                               dtype=batch["entities"].dtype)
            if t.start == 0:
                ent_acs[:, 1:, :self.args.n_agents] = (
                    batch["actions_onehot"][:, slice(0, t.stop - 1)])
            else:
                ent_acs[:, :, :self.args.n_agents] = (
                    batch["actions_onehot"][:, slice(t.start - 1, t.stop - 1)])
            entity_parts.append(ent_acs)
        entities = th.cat(entity_parts, dim=3)
        if imagine_inps is not None:
            imagine_inps['entities'] = entities.repeat(2, 1, 1, 1)
            imagine_inps['reset'] = imagine_inps['reset'].float()
        rets = {'entities': entities,
                'obs_mask': batch["obs_mask"][:, t],
                'entity_mask': batch["entity_mask"][:, t],
                'reset': batch["reset"][:, t].float()}
        if self.args.multi_task:
            rets['entity2task_mask'] = batch['entity2task_mask'][:, t]
            rets['task_mask'] = batch['task_mask'][:, t]
            if target:
                # if we're computing a bootstrapping target, use the task
                # assignments from the previous step
                task_t = slice(max(t.start - 1, 0), t.stop - 1)
                rets['entity2task_mask'] = batch['entity2task_mask'][:, task_t]
                if rets['entity2task_mask'].shape[1] < (t.stop - t.start):
                    rets['entity2task_mask'] = th.cat([rets['entity2task_mask'][:, [0]],
                                                       rets['entity2task_mask']],
                                                      dim=1)
        return rets, imagine_inps

    def _get_input_shape(self, scheme):
        agent_input_shape = scheme["entities"]["vshape"]
        if self.args.entity_last_action:
            agent_input_shape += scheme["actions_onehot"]["vshape"][0]
        hier_input_shape = scheme["entities"]["vshape"]
        return agent_input_shape, hier_input_shape

    def init_hidden(self, batch_size):
        super().init_hidden(batch_size)
        if self.use_alloc:
            self.task_allocations = th.zeros(
                batch_size, self.n_agents, self.args.n_tasks, device=self.agent._base.fc1.weight.device)
        if self.use_copa:
            self.coach_z = th.zeros(
                batch_size, self.n_agents, self.args.rnn_hidden_dim, device=self.agent._base.fc1.weight.device
            )

    def load_state(self, other_mac):
        super().load_state(other_mac)
        if self.use_copa:
            self.coach.load_state_dict(other_mac.coach.state_dict())
            if self.args.hier_agent['copa_vi_loss']:
                self.copa_recog.load_state_dict(other_mac.copa_recog.state_dict())

    def load_alloc_state(self, other_mac):
        if self.learned_alloc:
            self.alloc_policy.load_state_dict(other_mac.alloc_policy.state_dict())
            self.alloc_critic.load_state_dict(other_mac.alloc_critic.state_dict())

    def cuda(self):
        super().cuda()
        if self.learned_alloc:
            self.alloc_policy.cuda()
            self.alloc_critic.cuda()
        if self.use_copa:
            self.coach.cuda()
            if self.args.hier_agent['copa_vi_loss']:
                self.copa_recog.cuda()

    def eval(self):
        super().eval()
        if self.learned_alloc:
            self.alloc_policy.eval()
            self.alloc_critic.eval()
        if self.use_copa:
            self.coach.eval()
            if self.args.hier_agent['copa_vi_loss']:
                self.copa_recog.eval()

    def train(self):
        super().train()
        if self.learned_alloc:
            self.alloc_policy.train()
            self.alloc_critic.train()
        if self.use_copa:
            self.coach.train()
            if self.args.hier_agent['copa_vi_loss']:
                self.copa_recog.train()

    def save_models(self, path):
        super().save_models(path)
        if self.learned_alloc:
            th.save(self.alloc_policy.state_dict(), "{}alloc_pi.th".format(path))
            th.save(self.alloc_critic.state_dict(), "{}alloc_q.th".format(path))
        if self.use_copa:
            th.save(self.coach.state_dict(), "{}copa_coach.th".format(path))
            if self.args.hier_agent['copa_vi_loss']:
                th.save(self.copa_recog.state_dict(), "{}copa_recog.th".format(path))

    def load_models(self, path, pi_only=False):
        super().load_models(path)
        if pi_only:
            return
        if self.learned_alloc:
            self.alloc_policy.load_state_dict(th.load("{}alloc_pi.th".format(path), map_location=lambda storage, loc: storage))
            self.alloc_critic.load_state_dict(th.load("{}alloc_q.th".format(path), map_location=lambda storage, loc: storage))
        if self.use_copa:
            self.coach.load_state_dict(th.load("{}copa_coach.th".format(path), map_location=lambda storage, loc: storage))
            if self.args.hier_agent['copa_vi_loss']:
                self.copa_recog.load_state_dict(th.load("{}copa_recog.th".format(path), map_location=lambda storage, loc: storage))
