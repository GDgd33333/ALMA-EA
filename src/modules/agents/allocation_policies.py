import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical, RelaxedOneHotCategorical
from ..layers import EntityAttentionLayer
from .allocation_common import groupmask2attnmask, TaskEmbedder, COUNT_NORM_FACTOR

class AutoregressiveAllocPolicy(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args

        self.args = args
        self.pi_ag_attn = self.args.hier_agent['pi_ag_attn']
        self.pi_pointer_net = self.args.hier_agent['pi_pointer_net']
        self.subtask_mask = self.args.hier_agent['subtask_mask']
        self.sel_task_upd = self.args.hier_agent['sel_task_upd']
        self.pi_autoreg = self.args.hier_agent['pi_autoreg']

        embd_upd_in_shape = args.attn_embed_dim * 2
        if not self.pi_pointer_net:
            input_shape += args.n_tasks
            embd_upd_in_shape += args.n_tasks
        if not self.sel_task_upd:
            embd_upd_in_shape += args.attn_embed_dim
        self.fc1 = nn.Linear(input_shape, args.attn_embed_dim)
        self.task_embed = TaskEmbedder(args.attn_embed_dim, args)
        self.attn = EntityAttentionLayer(args.attn_embed_dim,
                                         args.attn_embed_dim,
                                         args.attn_embed_dim, args)
        if self.pi_autoreg:
            self.embed_update = nn.Linear(embd_upd_in_shape, args.attn_embed_dim)
            if self.pi_pointer_net:
                self.count_embed = nn.Linear(2, args.attn_embed_dim)
        elif self.pi_pointer_net:
            # can only embed nonagent entity counts since agent allocs are decided all at once
            self.count_embed = nn.Linear(1, args.attn_embed_dim)
        if not self.pi_pointer_net:
            self.out_fc = nn.Linear(args.attn_embed_dim * 2, args.n_tasks)
        self.register_buffer('sample_temp',
                             th.scalar_tensor(1))  # TODO: anneal or try other values?
        self.register_buffer('scale_factor',
                             th.scalar_tensor(args.attn_embed_dim).sqrt())

    @property
    def device(self):
        return self.fc1.weight.device

    def _normalize_entity_mask(self, entity_mask, batch_size):
        """
        规范化entity_mask的维度，确保它是2维的 (bs, n_entities)
        
        Args:
            entity_mask: 可能是1维 (n_entities,) 或2维 (bs, n_entities) 的张量
            batch_size: 目标batch size
            
        Returns:
            entity_mask: 2维张量 (bs, n_entities)
        """
        if entity_mask.dim() == 1:
            # 如果是1维 (n_entities,)，扩展为 (1, n_entities) 然后repeat到batch size
            entity_mask = entity_mask.unsqueeze(0).repeat(batch_size, 1)  # (bs, n_entities)
        elif entity_mask.dim() == 2:
            # 如果已经是2维，检查batch size是否匹配
            if entity_mask.shape[0] == 1 and batch_size > 1:
                # 如果是 (1, n_entities) 但需要更大的batch size，则repeat
                entity_mask = entity_mask.repeat(batch_size, 1)  # (bs, n_entities)
            elif entity_mask.shape[0] != batch_size:
                # 如果batch size不匹配，可能需要调整（这里假设是单样本需要扩展）
                if batch_size > entity_mask.shape[0]:
                    # 扩展batch size
                    repeat_times = batch_size // entity_mask.shape[0]
                    remainder = batch_size % entity_mask.shape[0]
                    entity_mask = th.cat([
                        entity_mask.repeat(repeat_times, 1),
                        entity_mask[:remainder]
                    ], dim=0) if remainder > 0 else entity_mask.repeat(repeat_times, 1)
        return entity_mask

    def _autoreg_forward(self, task_embeds, task_nonag_counts, agent_embeds, task_mask, entity_mask, avail_actions,
                         calc_stats=False, test_mode=False, repeat_fn=lambda x: x):
        nt = self.args.n_tasks
        bs, na, _ = agent_embeds.shape
        stats = {}
        task_embeds = repeat_fn(task_embeds)
        task_mask = repeat_fn(task_mask)
        if task_nonag_counts is not None:
            task_nonag_counts = repeat_fn(task_nonag_counts)
        prop_bs = task_mask.shape[0]
        allocs = repeat_fn(th.zeros((bs, na, nt), device=agent_embeds.device))
        all_log_pi = th.zeros_like(allocs)

        task_ag_counts = th.zeros_like(allocs[:, 0])

        # 规范化entity_mask维度
        entity_mask = self._normalize_entity_mask(entity_mask, bs)
        agent_mask = entity_mask[:, :na]

        prev_alloc_mask = th.zeros_like(task_mask)

        for ai in range(self.args.n_agents):
            # compute pointer-net logits (scale as in dot-product attention)
            curr_agent_embed = agent_embeds[:, [ai]]
            curr_agent_embed = repeat_fn(curr_agent_embed)

            if self.pi_pointer_net:
                count_embeds = self.count_embed(th.stack([task_nonag_counts, task_ag_counts], dim=-1))
                logits = th.bmm(curr_agent_embed, (task_embeds + count_embeds).transpose(1, 2)).squeeze(1) / self.scale_factor
            else:
                # curr_agent_embed.shape = (bs, 1, hd), task_embeds.shape = (bs, hd)
                logit_ins = th.cat([curr_agent_embed.squeeze(1), task_embeds], dim=1)
                logits = self.out_fc(F.relu(logit_ins))

            # mask inactive tasks s.t. softmax is 0（关键修复：使用-1e9而不是finfo.min）
            NEG = -1e9
            curr_mask = task_mask.clone()
            masked_logits = logits.masked_fill(curr_mask.bool(), NEG)
            # mask for inactive agents
            curr_agent_mask = repeat_fn(1 - agent_mask[:, [ai]].float())
            dist = RelaxedOneHotCategorical(self.sample_temp, logits=masked_logits)
            # sample action
            soft_ac = dist.rsample()
            if calc_stats:
                # NOTE: we can use dist.logits as log prob as pytorch
                # Categorical distribution normalizes logits such that
                # th.exp(dist.logits) is the probability
                all_log_pi[:, ai] = dist.logits
                ag_log_pi = dist.logits.gather(1, soft_ac.argmax(dim=1, keepdim=True))
                stats['log_pi'] = stats.get('log_pi', 0) + ag_log_pi * curr_agent_mask
                stats['best_prob'] = stats.get('best_prob', 0) + dist.probs.max(dim=1, keepdim=True)[0] * curr_agent_mask
                entropy = -(dist.logits * dist.probs).sum(dim=1, keepdim=True)
                stats['entropy'] = stats.get('entropy', 0) + entropy * curr_agent_mask
            # make one-hot sample that acts like a continuous sample in the backward pass
            onehot_ac = F.one_hot(soft_ac.argmax(dim=1), num_classes=nt).float()
            hard_ac = onehot_ac - soft_ac.detach() + soft_ac
            hard_ac = hard_ac * curr_agent_mask
            prev_alloc_mask += hard_ac.detach().to(th.uint8)
            task_ag_counts += hard_ac.detach() * COUNT_NORM_FACTOR
            allocs[:, ai] = hard_ac
            # update embedding of selected task to incorporate new agent (only if agent is active)
            if self.pi_pointer_net:
                if self.sel_task_upd:
                    # only update selected task embeddings
                    embed_upd_in = th.cat([task_embeds, curr_agent_embed.repeat(1, nt, 1)], dim=2)
                    task_embeds = task_embeds + self.embed_update(F.relu(embed_upd_in)) * hard_ac.detach().unsqueeze(2) * curr_agent_mask.unsqueeze(2)
                else:
                    # update all task embeddings (use copy of selected task to condition on the previous agents' allocations)
                    sel_task_embed = (task_embeds * hard_ac.detach().unsqueeze(2)).sum(dim=1, keepdim=True)
                    embed_upd_in = th.cat([task_embeds, curr_agent_embed.repeat(1, nt, 1), sel_task_embed.repeat(1, nt, 1)], dim=2)
                    task_embeds = task_embeds + self.embed_update(F.relu(embed_upd_in)) * curr_agent_mask.unsqueeze(2)
            else:
                embed_upd_in = th.cat([task_embeds, curr_agent_embed.squeeze(1), hard_ac.detach()], dim=1)
                task_embeds = task_embeds + self.embed_update(F.relu(embed_upd_in)) * curr_agent_mask
        if calc_stats:
            stats['all_log_pi'] = all_log_pi
        return allocs, stats

    def _standard_forward(self, task_embeds, task_nonag_counts, agent_embeds, task_mask, entity_mask,
                         calc_stats=False, test_mode=False, repeat_fn=lambda x: x):
        nt = self.args.n_tasks
        bs, na, _ = agent_embeds.shape
        stats = {}
        task_embeds = repeat_fn(task_embeds) + repeat_fn(self.count_embed(task_nonag_counts.unsqueeze(-1)))
        task_mask = repeat_fn(task_mask)
        agent_embeds = repeat_fn(agent_embeds)
        allocs = repeat_fn(th.zeros((bs, na, nt), device=agent_embeds.device))

        if self.pi_pointer_net:
            logits = th.bmm(agent_embeds, task_embeds.transpose(1, 2)) / self.scale_factor
        else:
            raise NotImplementedError
            # curr_agent_embed.shape = (bs, na, hd), task_embeds.shape(bs, nt, hd)
            logit_ins = th.cat([agent_embeds.unsqueeze(2).repeat(1, 1, nt, 1),
                                task_embeds.unsqueeze(1).repeat(1, na, 1, 1)], dim=3)
            logits = self.out_fc(F.relu(logit_ins)).squeeze(3)

        # mask inactive tasks s.t. softmax is 0
        # 关键修复：使用-1e9而不是finfo.min
        NEG = -1e9
        masked_logits = logits.masked_fill(task_mask.unsqueeze(1).bool(), NEG)
        # mask for inactive agents
        # 规范化entity_mask维度
        entity_mask = self._normalize_entity_mask(entity_mask, bs)
        agent_mask = repeat_fn(1 - entity_mask[:, :na].float())
        dist = RelaxedOneHotCategorical(self.sample_temp, logits=masked_logits)
        # sample action
        soft_ac = dist.rsample()
        if calc_stats:
            # NOTE: we can use dist.logits as log prob as pytorch
            # Categorical distribution normalizes logits such that
            # th.exp(dist.logits) is the probability
            stats['all_log_pi'] = dist.logits
            ag_log_pi = dist.logits.gather(2, soft_ac.argmax(dim=2, keepdim=True)).squeeze(2)
            stats['log_pi'] = (ag_log_pi * agent_mask).sum(dim=1, keepdim=True)
            stats['best_prob'] = (dist.probs.max(dim=2)[0] * agent_mask).sum(dim=1, keepdim=True)
            entropy = -(dist.logits * dist.probs).sum(dim=2)
            stats['entropy'] = (entropy * agent_mask).sum(dim=1, keepdim=True)
        # make one-hot sample that acts like a continuous sample in the backward pass
        onehot_ac = F.one_hot(soft_ac.argmax(dim=2), num_classes=nt).float()
        allocs = onehot_ac - soft_ac.detach() + soft_ac
        allocs = allocs * agent_mask.unsqueeze(2)
        return allocs, stats

    def forward(self, batch, calc_stats=False, test_mode=False, n_proposals=-1):
        # copy entity2task mask and zero out assignments
        entities = batch['entities']
        entity_mask = batch['entity_mask']
        entity2task_mask = batch['entity2task_mask']
        avail_actions = batch['avail_actions']

        # 规范化entity_mask维度
        bs = entities.shape[0]
        entity_mask = self._normalize_entity_mask(entity_mask, bs)

        nag = self.args.n_agents
        entity2task = 1 - entity2task_mask.float()
        last_alloc = batch['last_alloc']
        entity2task[:, :nag] = last_alloc

        # observe which task agents were assigned to in previous step + which
        # task non-agent entities belong to
        if not self.pi_pointer_net:
            entities = th.cat([entities, entity2task], dim=-1)
        x1 = self.fc1(entities)
        if self.pi_pointer_net:
            x1 += self.task_embed(entity2task)

        # compute attention for non-agent entities and get embedding for each task
        nonagent_x1 = x1[:, nag:]
        if self.pi_pointer_net and self.subtask_mask:
            nonagent_attn_mask = groupmask2attnmask(
                entity2task_mask[:, nag:])
        else:
            nonagent_attn_mask = groupmask2attnmask(
                entity_mask[:, nag:])
        nonagent_mask = entity_mask[:, nag:]
        nonagent_x2 = self.attn(F.relu(nonagent_x1), pre_mask=nonagent_attn_mask,
                                post_mask=nonagent_mask)
        if self.pi_pointer_net:
            nonagent_entity2task = entity2task[:, nag:]  # (bs, n_nonagent, nt)
            # sum up embeddings of non-agent entities belonging to each task
            task_x2 = th.bmm(nonagent_entity2task.transpose(1, 2), nonagent_x2)
            # count nonagent entities present in each task
            task_nonag_cnt = nonagent_entity2task.sum(dim=1) * COUNT_NORM_FACTOR
        else:
            task_x2 = nonagent_x2.mean(dim=1)
            task_nonag_cnt = None

        # get agent embeddings
        agent_x1 = x1[:, :nag]
        if self.pi_ag_attn:
            ag_mask = entity_mask[:, :nag]
            active_mask = groupmask2attnmask(ag_mask)
            inverse_causal_mask = th.diag(
                th.ones(nag, device=ag_mask.device)
            ).cumsum(dim=1).transpose(0,1).to(th.uint8)
            ag_attn_mask = (active_mask + inverse_causal_mask).min(th.ones_like(active_mask))
            agent_embeds = agent_x1 + self.attn(
                F.relu(agent_x1), pre_mask=ag_attn_mask, post_mask=ag_mask)
        else:
            agent_embeds = agent_x1

        repeat = 1
        if n_proposals > 0:
            repeat = n_proposals
        repeat_fn = lambda x: x.repeat_interleave(repeat, dim=0)

        if self.pi_autoreg:
            allocs, stats = self._autoreg_forward(
                task_x2, task_nonag_cnt, agent_embeds, batch['task_mask'], entity_mask, avail_actions,
                calc_stats=calc_stats, test_mode=test_mode, repeat_fn=repeat_fn)
        else:
            allocs, stats = self._standard_forward(
                task_x2, task_nonag_cnt, agent_embeds, batch['task_mask'], entity_mask,
                calc_stats=calc_stats, test_mode=test_mode, repeat_fn=repeat_fn)

        if n_proposals > 1:
            allocs = allocs.reshape(-1, n_proposals, nag, self.args.n_tasks)
        if calc_stats:
            stats['best_prob'] = stats['best_prob'] / repeat_fn(1 - entity_mask[:, :nag].float()).sum(dim=1, keepdim=True)
            for k, v in stats.items():
                stats[k] = v.reshape(-1, n_proposals, *v.shape[1:])
            return allocs, stats
        return allocs


class A2CAllocPolicy(nn.Module):
    """
    A2C任务分配Actor网络
    
    这个网络用于A2C算法中的高层任务分配决策。它将环境状态（实体信息）映射为任务分配动作。
    
    特点：
    1. 直接输出单个动作（不再生成多个提案，与AQL不同）
    2. 返回动作、log概率和熵（用于策略梯度计算）
    3. 复用ALMA的特征提取逻辑（实体编码、注意力机制等）
    4. 使用标准Categorical分布进行采样（保证log_prob与采样动作一致）
    
    网络结构：
    输入: entities (实体特征) -> 特征提取 -> 任务嵌入 + 智能体嵌入 -> 动作logits -> Categorical采样 -> 分配动作
    """
    def __init__(self, input_shape, args):
        """
        初始化A2C分配策略网络
        
        Args:
            input_shape: 输入特征维度（实体特征的维度）
            args: 配置参数对象，包含网络超参数
        """
        super().__init__()
        self.args = args
        
        # ==================== 特征提取配置 ====================
        # 这些配置控制特征提取的方式，复用ALMA的设计
        
        # pi_pointer_net: 是否使用指针网络（点积注意力）计算logits
        # True: 使用点积注意力（更高效，适合任务数量多的情况）
        # False: 使用全连接层（需要将entity2task信息concat到输入）
        self.pi_pointer_net = self.args.hier_agent.get('pi_pointer_net', True)
        
        # subtask_mask: 是否使用子任务掩码进行注意力计算
        # True: 只允许同一任务的实体之间相互关注（更精确的任务特征聚合）
        # False: 所有实体都可以相互关注（更通用的特征聚合）
        self.subtask_mask = self.args.hier_agent.get('subtask_mask', True)
        
        # pi_ag_attn: 是否对智能体使用自注意力机制
        # True: 智能体之间可以相互关注（建模智能体间的协作关系）
        # False: 每个智能体独立编码（更简单，计算更快）
        self.pi_ag_attn = self.args.hier_agent.get('pi_ag_attn', False)
        
        # ==================== 输入维度调整 ====================
        # 如果不使用指针网络，需要将entity2task信息直接concat到输入特征中
        if not self.pi_pointer_net:
            input_shape += args.n_tasks  # 增加n_tasks维（每个任务一个维度）
        
        # ==================== 基础特征提取层 ====================
        # fc1: 将原始实体特征映射到嵌入空间
        # 输入: (bs, n_entities, input_shape) -> 输出: (bs, n_entities, attn_embed_dim)
        self.fc1 = nn.Linear(input_shape, args.attn_embed_dim)
        
        # task_embed: 任务嵌入层（将entity2task映射转换为任务相关的嵌入）
        # 用于增强实体特征，使其包含任务分配信息
        self.task_embed = TaskEmbedder(args.attn_embed_dim, args)
        
        # attn: 实体注意力层（用于聚合实体特征）
        # 用于非智能体实体：聚合同一任务的实体特征，得到任务嵌入
        # 用于智能体（如果pi_ag_attn=True）：智能体之间的相互关注
        self.attn = EntityAttentionLayer(
            args.attn_embed_dim,  # 输入维度
            args.attn_embed_dim,  # 输出维度
            args.attn_embed_dim,  # 注意力维度
            args
        )
        
        # ==================== 编码器层 ====================
        # 对提取的特征进行进一步编码，增强表达能力
        
        # agent_encoder: 智能体编码器
        # 对每个智能体的嵌入进行编码，增强智能体特征的表达能力
        # 使用LayerNorm稳定训练，ReLU增加非线性
        self.agent_encoder = nn.Sequential(
            nn.Linear(args.attn_embed_dim, args.attn_embed_dim),  # 线性变换
            nn.LayerNorm(args.attn_embed_dim),  # 层归一化（稳定训练）
            nn.ReLU()  # 非线性激活
        )
        
        # task_encoder: 任务编码器
        # 对每个任务的嵌入进行编码，增强任务特征的表达能力
        # 结构与agent_encoder相同
        self.task_encoder = nn.Sequential(
            nn.Linear(args.attn_embed_dim, args.attn_embed_dim),
            nn.LayerNorm(args.attn_embed_dim),
            nn.ReLU()
        )
        
        # ==================== 动作输出层 ====================
        # ==================== 自回归AR相关模块（原版ALMA方式）====================
        # 原版ALMA使用动态更新task_embeds来传递已分配agent的信息
        # 而不是使用prefix特征
        
        # sel_task_upd: 是否只更新被选中的任务embedding
        # True: 只更新被选中的任务（更精确，但计算稍慢）
        # False: 更新所有任务（更通用，计算更快）
        self.sel_task_upd = self.args.hier_agent.get('sel_task_upd', True)
        
        # embed_update: 用于动态更新task_embeds的模块
        # 输入维度取决于pi_pointer_net和sel_task_upd
        embd_upd_in_shape = args.attn_embed_dim * 2  # task_embeds + agent_embed
        if not self.pi_pointer_net:
            embd_upd_in_shape += args.n_tasks  # 如果不使用指针网络，需要concat one-hot action
        if not self.sel_task_upd:
            embd_upd_in_shape += args.attn_embed_dim  # 如果更新所有任务，需要concat selected_task_embed
        
        self.embed_update = nn.Linear(embd_upd_in_shape, args.attn_embed_dim)
        
        # count_embed: 用于编码任务分配计数（类似原版ALMA）
        # 输入: (bs, nt, 2) - [task_nonag_counts, task_ag_counts]
        # 输出: (bs, nt, embed_dim) - 计数embedding
        self.count_embed = nn.Linear(2, args.attn_embed_dim)
        
        # action_head: 将(智能体, 任务)对的组合特征映射为logit值
        # 对于AR模式，使用指针网络（点积注意力）或全连接层
        if self.pi_pointer_net:
            # 指针网络：使用点积注意力计算logits
            # 不需要action_head，直接使用点积
            self.action_head = None
        else:
            # 全连接层：将组合特征映射为logit
            self.action_head = nn.Sequential(
                nn.Linear(args.attn_embed_dim * 2, args.attn_embed_dim),
                nn.ReLU(),
                nn.Linear(args.attn_embed_dim, 1)
            )
        
        # action_head_parallel: 并行版本的action_head（不包含prefix）
        # 用于非AR模式，保持向后兼容
        self.action_head_parallel = nn.Sequential(
            nn.Linear(args.attn_embed_dim * 2, args.attn_embed_dim),
            nn.ReLU(),
            nn.Linear(args.attn_embed_dim, 1)
        )
        
        # scale_factor: 用于指针网络的缩放（类似原版ALMA）
        self.register_buffer('scale_factor', th.scalar_tensor(args.attn_embed_dim).sqrt())
    
    def _compute_logits_parallel(self, agent_embeds, task_embeds, batch):
        """
        计算并行模式的logits（公共函数，确保forward和compute_logprob_for_actions一致）
        
        Args:
            agent_embeds: (bs, na, embed_dim) - 智能体嵌入
            task_embeds: (bs, nt, embed_dim) - 任务嵌入
            batch: 包含状态信息的字典
        
        Returns:
            masked_logits: (bs, na, nt) - 应用mask后的logits
            agent_task_mask: (bs, na, nt) - agent-task mask（用于后续检查）
        """
        bs, na, _ = agent_embeds.shape
        nt = self.args.n_tasks
        
        # 组合特征
        agent_expanded = agent_embeds.unsqueeze(2).repeat(1, 1, nt, 1)  # (bs, na, nt, embed_dim)
        task_expanded = task_embeds.unsqueeze(1).repeat(1, na, 1, 1)  # (bs, na, nt, embed_dim)
        combined = th.cat([agent_expanded, task_expanded], dim=-1)  # (bs, na, nt, embed_dim*2)
        
        # 计算logits
        if self.pi_pointer_net:
            # 指针网络：使用点积注意力
            # agent_embeds: (bs, na, d), task_embeds: (bs, nt, d)
            # 使用einsum计算所有(agent, task)对的点积
            logits = th.einsum('bad,btd->bat', agent_embeds, task_embeds) / self.scale_factor  # (bs, na, nt)
        else:
            # 全连接层
            logits = self.action_head_parallel(combined).squeeze(-1)  # (bs, na, nt)
        
        # 限制logits范围
        logits = th.clamp(logits, min=-50.0, max=50.0)
        
        # 统一mask处理：task_mask + avail_actions
        task_mask = batch['task_mask']  # (bs, nt)
        
        # 构建agent_task_mask
        if 'avail_actions' in batch and batch['avail_actions'] is not None:
            # 优先使用avail_actions（per-agent mask）
            avail_actions = batch['avail_actions']  # (bs, na, nt) 或 (bs, nt)
            if avail_actions.dim() == 2:
                avail_actions = avail_actions.unsqueeze(1).expand(-1, na, -1)  # (bs, na, nt)
            agent_task_mask = 1 - avail_actions.float()  # (bs, na, nt) - 1表示不可用，0表示可用
        else:
            # 使用全局task_mask
            if task_mask.dim() == 2:
                agent_task_mask = task_mask.unsqueeze(1).expand(-1, na, -1)  # (bs, na, nt)
            else:
                agent_task_mask = task_mask  # 已经是(bs, na, nt)
        
        # 应用mask
        NEG = -1e9
        masked_logits = logits.masked_fill(agent_task_mask.bool(), NEG)
        
        return masked_logits, agent_task_mask
        
        # 注意：不再需要sample_temp（采样温度），因为使用标准Categorical分布
        # RelaxedOneHotCategorical需要温度参数，但Categorical不需要
    
    @property
    def device(self):
        return self.fc1.weight.device
    
    def forward(self, batch, calc_stats=False, test_mode=False):
        """
        A2C Actor前向传播
        
        完整流程：
        1. 特征提取：从实体特征中提取任务嵌入和智能体嵌入
        2. 动作生成：计算每个智能体对每个任务的logits，并采样动作
        3. 统计计算（可选）：计算log概率和熵，用于策略梯度
        
        Args:
            batch: 包含以下键的字典
                - 'entities': (bs, n_entities, entity_dim) - 实体特征
                - 'entity_mask': (bs, n_entities) - 实体掩码（1表示无效，0表示有效）
                - 'entity2task_mask': (bs, n_entities, n_tasks) - 实体到任务的掩码
                - 'task_mask': (bs, n_tasks) - 任务掩码（1表示无效任务，0表示有效任务）
                - 'last_alloc': (bs, n_agents, n_tasks) - 上次的分配结果（可选）
            calc_stats: 是否计算统计信息（log_prob和entropy），用于训练
            test_mode: 是否为测试模式
                - True: 使用greedy策略（选择概率最高的动作）
                - False: 使用采样策略（从分布中采样）
        
        Returns:
            actions: (bs, n_agents, n_tasks) - 任务分配动作（one-hot编码）
            stats: 字典（仅在calc_stats=True时返回），包含：
                - 'log_prob': (bs, 1) - 联合log概率（所有智能体的log概率之和）
                - 'entropy': (bs, 1) - 联合熵（所有智能体的熵之和）
                - 'log_probs_per_agent': (bs, n_agents) - 每个智能体的log概率
                - 'action': (bs, n_agents, n_tasks) - 分配动作（与返回的actions相同）
        """
        # ==================== 输入提取和验证 ====================
        entities = batch['entities']  # (bs, n_entities, entity_dim) - 实体特征
        entity_mask = batch['entity_mask']  # (bs, n_entities) - 实体掩码
        task_mask = batch['task_mask']  # (bs, n_tasks) - 任务掩码
        
        # 检查输入数据是否有NaN（早期检查，防止NaN传播）
        # 如果发现NaN，用0替换（避免训练崩溃）
        if th.isnan(entities).any():
            print(f"ERROR: NaN in input entities! Replacing with zeros.")
            entities = th.where(th.isnan(entities), th.zeros_like(entities), entities)
        
        # 获取维度信息
        na = self.args.n_agents  # 智能体数量
        nt = self.args.n_tasks  # 任务数量
        bs = entities.shape[0]  # batch size（批次大小）
        
        # ==================== 第一步：特征提取 ====================
        # 这部分复用ALMA的特征提取逻辑，将原始实体特征转换为嵌入表示
        
        # 1.1 处理实体到任务的映射
        # entity2task_mask: (bs, n_entities, n_tasks)，1表示"不属于该任务"，0表示"属于该任务"
        # entity2task: 转换为0/1表示，1表示"属于该任务"，0表示"不属于该任务"
        entity2task = 1 - batch['entity2task_mask'].float()  # (bs, n_entities, n_tasks)
        
        # 1.2 处理智能体的上次分配结果
        # last_alloc: (bs, n_agents, n_tasks) - 上次决策的分配结果（one-hot编码）
        # 如果batch中没有last_alloc，则初始化为全0（表示没有上次分配）
        last_alloc = batch.get('last_alloc', th.zeros_like(entity2task[:, :na]))
        
        # 将智能体的entity2task替换为last_alloc（智能体的分配来自上次决策，不是环境状态）
        # 这样可以让网络知道"上次将智能体分配到了哪个任务"
        entity2task[:, :na] = last_alloc  # 更新智能体的任务分配信息
        
        # 1.3 根据是否使用指针网络，决定如何将任务信息融入输入
        if not self.pi_pointer_net:
            # 如果不使用指针网络，需要将entity2task信息直接concat到输入特征
            # 这样网络可以直接看到"每个实体属于哪些任务"
            entities = th.cat([entities, entity2task], dim=-1)  # (bs, n_entities, entity_dim + n_tasks)
        
        # 1.4 基础特征编码
        # 将实体特征（可能包含entity2task信息）映射到嵌入空间
        x1 = self.fc1(entities)  # (bs, n_entities, attn_embed_dim)
        
        # 检查fc1输出是否有NaN（防止NaN传播）
        if th.isnan(x1).any():
            print(f"ERROR: NaN after fc1! Check network parameters.")
            x1 = th.where(th.isnan(x1), th.zeros_like(x1), x1)
        
        # 1.5 任务嵌入增强（如果使用指针网络）
        if self.pi_pointer_net:
            # 使用TaskEmbedder将entity2task映射转换为任务相关的嵌入
            # 这个嵌入会被加到实体特征上，增强特征的任务相关性
            task_embed_out = self.task_embed(entity2task)  # (bs, n_entities, attn_embed_dim)
            
            # 检查task_embed输出是否有NaN
            if th.isnan(task_embed_out).any():
                print(f"ERROR: NaN after task_embed! Check network parameters.")
                task_embed_out = th.where(th.isnan(task_embed_out), th.zeros_like(task_embed_out), task_embed_out)
            
            # 将任务嵌入加到实体特征上（残差连接的思想）
            x1 += task_embed_out  # (bs, n_entities, attn_embed_dim)
        
        # ==================== 第二步：处理非智能体实体（任务相关实体）====================
        # 非智能体实体（如任务目标、资源点等）属于特定任务，需要聚合为任务特征
        
        # 2.1 提取非智能体实体的特征和掩码
        nonagent_x1 = x1[:, na:]  # (bs, n_nonagents, attn_embed_dim) - 非智能体实体特征
        nonagent_mask = entity_mask[:, na:]  # (bs, n_nonagents) - 非智能体实体掩码
        
        # 2.2 构建注意力掩码
        # 注意力掩码控制哪些实体之间可以相互关注
        if self.pi_pointer_net and self.subtask_mask:
            # 如果使用子任务掩码：只允许同一任务的实体之间相互关注
            # 这样可以更精确地聚合任务特征（同一任务的实体特征会聚合在一起）
            nonagent_attn_mask = groupmask2attnmask(
                batch['entity2task_mask'][:, na:]  # (bs, n_nonagents, n_tasks)
            )
        else:
            # 如果不使用子任务掩码：所有实体都可以相互关注（更通用的聚合）
            nonagent_attn_mask = groupmask2attnmask(nonagent_mask)
        
        # 2.3 非智能体实体注意力聚合
        # 使用注意力机制聚合非智能体实体特征
        # 如果使用subtask_mask，同一任务的实体会聚合在一起
        nonagent_x2 = self.attn(
            F.relu(nonagent_x1),  # 输入特征（ReLU激活）
            pre_mask=nonagent_attn_mask,  # 注意力掩码（控制哪些实体可以关注）
            post_mask=nonagent_mask  # 输出掩码（过滤无效实体）
        )  # (bs, n_nonagents, attn_embed_dim)
        
        # 2.4 聚合任务特征
        # 将属于同一任务的非智能体实体特征聚合，得到每个任务的特征嵌入
        if self.pi_pointer_net:
            # 使用entity2task作为权重，对非智能体实体特征进行加权求和
            # 每个任务的嵌入 = 属于该任务的所有非智能体实体特征的加权和
            nonagent_entity2task = entity2task[:, na:]  # (bs, n_nonagents, n_tasks)
            task_embeds = th.bmm(
                nonagent_entity2task.transpose(1, 2),  # (bs, n_tasks, n_nonagents) - 转置后作为权重
                nonagent_x2  # (bs, n_nonagents, attn_embed_dim) - 非智能体实体特征
            )  # (bs, n_tasks, attn_embed_dim) - 每个任务的特征嵌入
        else:
            # 如果不使用指针网络：简单地对所有非智能体实体特征取平均
            # 然后复制给所有任务（所有任务共享相同的特征）
            task_embeds = nonagent_x2.mean(dim=1, keepdim=True).repeat(1, nt, 1)  # (bs, n_tasks, attn_embed_dim)
        
        # 2.5 任务特征编码
        # 使用任务编码器进一步处理任务特征，增强表达能力
        task_embeds = self.task_encoder(task_embeds)  # (bs, n_tasks, attn_embed_dim)
        
        # 检查task_encoder输出是否有NaN
        if th.isnan(task_embeds).any():
            print(f"ERROR: NaN after task_encoder! Check network parameters.")
            task_embeds = th.where(th.isnan(task_embeds), th.zeros_like(task_embeds), task_embeds)
        
        # ==================== 第三步：处理智能体特征 ====================
        # 提取并编码每个智能体的特征嵌入
        
        # 3.1 提取智能体特征
        agent_x1 = x1[:, :na]  # (bs, n_agents, attn_embed_dim) - 智能体特征
        
        # 3.2 智能体自注意力（可选）
        if self.pi_ag_attn:
            # 如果启用智能体自注意力：智能体之间可以相互关注
            # 这样可以建模智能体之间的协作关系（例如：智能体A知道智能体B的能力）
            ag_mask = entity_mask[:, :na]  # (bs, n_agents) - 智能体掩码
            active_mask = groupmask2attnmask(ag_mask)  # 构建注意力掩码
            
            # 智能体自注意力：每个智能体可以关注其他智能体
            agent_attn_out = self.attn(
                F.relu(agent_x1),  # 输入特征
                pre_mask=active_mask,  # 注意力掩码
                post_mask=ag_mask  # 输出掩码
            )  # (bs, n_agents, attn_embed_dim)
            
            # 检查attention输出是否有NaN
            if th.isnan(agent_attn_out).any():
                print(f"ERROR: NaN after agent attention! Check network parameters.")
                agent_attn_out = th.where(th.isnan(agent_attn_out), th.zeros_like(agent_attn_out), agent_attn_out)
            
            # 残差连接：原始特征 + 注意力输出
            agent_embeds = agent_x1 + agent_attn_out  # (bs, n_agents, attn_embed_dim)
        else:
            # 如果不使用智能体自注意力：每个智能体独立编码（不关注其他智能体）
            agent_embeds = agent_x1  # (bs, n_agents, attn_embed_dim)
        
        # 3.3 智能体特征编码
        # 使用智能体编码器进一步处理智能体特征，增强表达能力
        agent_embeds = self.agent_encoder(agent_embeds)  # (bs, n_agents, attn_embed_dim)
        
        # 检查agent_encoder输出是否有NaN
        if th.isnan(agent_embeds).any():
            print(f"ERROR: NaN after agent_encoder! Check network parameters.")
            agent_embeds = th.where(th.isnan(agent_embeds), th.zeros_like(agent_embeds), agent_embeds)
        
        # ==================== 第四步：生成动作logits ====================
        # 使用公共函数计算logits，确保与compute_logprob_for_actions一致
        masked_logits, agent_task_mask = self._compute_logits_parallel(agent_embeds, task_embeds, batch)
        
        # 检查是否有有效任务（防止全被mask的情况）
        has_valid_task = (~agent_task_mask.bool()).any(dim=-1)  # (bs, na) - 每个agent是否有有效任务
        
        # ==================== 第五步：采样动作 ====================
        # 向量化采样：一次性构建分布并采样，保证一致性
        
        # 5.1 处理全被mask的情况
        # 对于没有有效任务的agent，直接设为全0动作
        actions = th.zeros((bs, na, nt), device=masked_logits.device, dtype=th.float32)
        
        # 5.2 批量采样（只对有有效任务的agent采样）
        # 一次性构建所有agent的分布
        dist = Categorical(logits=masked_logits)  # (bs, na, nt) -> 自动广播为每个agent一个分布
        
        if test_mode:
            sampled_indices = dist.probs.argmax(dim=-1)  # (bs, na) - greedy
        else:
            sampled_indices = dist.sample()  # (bs, na) - sample
            
        # 向量化转换为one-hot
        actions = F.one_hot(sampled_indices, num_classes=nt).float()  # (bs, na, nt)
        
        # 对于没有有效任务的agent，动作设为全0
        actions = actions * has_valid_task.unsqueeze(-1)  # (bs, na, nt)
        
        # 5.3 应用智能体掩码
        # entity_mask: 1表示无效智能体，0表示有效智能体
        # agent_mask: 1表示有效智能体，0表示无效智能体（与entity_mask相反）
        agent_mask = (1 - entity_mask[:, :na].float()).unsqueeze(-1)  # (bs, na, 1)
        
        # 将无效智能体的动作设为0（无效智能体不参与分配）
        actions = actions * agent_mask  # (bs, na, nt)
        
        # ==================== 第六步：计算统计信息（可选）====================
        # 如果calc_stats=True，计算log概率和熵，用于策略梯度训练
        
        if calc_stats:
            # 6.1 准备智能体掩码（用于后续的加权求和）
            agent_mask_expanded = agent_mask.squeeze(-1)  # (bs, na)
            
            # 6.2 向量化计算log_prob和entropy（使用同一个dist，保证一致性）
            # 使用之前创建的dist，保证与采样时完全一致
            log_probs_per_agent = dist.log_prob(sampled_indices)  # (bs, na) - 向量化计算
            entropy_per_agent = dist.entropy()  # (bs, na) - 向量化计算
            
            # 对于没有有效任务的agent，log_prob和entropy设为0
            valid_mask = agent_mask_expanded * has_valid_task  # (bs, na) - 有效agent且有有效任务
            log_probs_per_agent = log_probs_per_agent * valid_mask
            entropy_per_agent = entropy_per_agent * valid_mask
            
            # 6.3 NaN检查和修复
            if th.isnan(log_probs_per_agent).any():
                if not hasattr(self, '_log_prob_nan_count'):
                    self._log_prob_nan_count = 0
                self._log_prob_nan_count += 1
                if self._log_prob_nan_count <= 1 or self._log_prob_nan_count % 100 == 0:
                    print(f"Warning: NaN in log_probs_per_agent (count: {self._log_prob_nan_count}). Replacing with zeros.")
                log_probs_per_agent = th.where(
                    th.isnan(log_probs_per_agent),
                    th.zeros_like(log_probs_per_agent),
                    log_probs_per_agent
                )
            
            # 6.4 计算联合log概率和熵
            joint_log_prob = log_probs_per_agent.sum(dim=1, keepdim=True)  # (bs, 1)
            joint_entropy = entropy_per_agent.sum(dim=1, keepdim=True)  # (bs, 1)
            
            # 最终检查
            if th.isnan(joint_log_prob).any():
                if not hasattr(self, '_joint_log_prob_nan_count'):
                    self._joint_log_prob_nan_count = 0
                self._joint_log_prob_nan_count += 1
                if self._joint_log_prob_nan_count <= 1 or self._joint_log_prob_nan_count % 100 == 0:
                    print(f"Warning: NaN in joint_log_prob (count: {self._joint_log_prob_nan_count}). Replacing with zeros.")
                joint_log_prob = th.where(
                    th.isnan(joint_log_prob),
                    th.zeros_like(joint_log_prob),
                    joint_log_prob
                )
            
            # 6.6 构建统计信息字典
            stats = {
                'log_prob': joint_log_prob,  # (bs, 1) - 联合log概率，用于策略梯度
                'entropy': joint_entropy,  # (bs, 1) - 联合熵，用于熵正则化
                'log_probs_per_agent': log_probs_per_agent,  # (bs, na) - 每个智能体的log概率（用于调试）
                'action': actions  # (bs, na, nt) - 分配动作（与返回的actions相同）
            }
            return actions, stats
        
        # 如果不需要统计信息，直接返回动作
        return actions
    
    def compute_logprob_for_actions(self, batch, actions_fixed):
        """
        计算固定actions的log_prob（用于PPO训练）
        
        Args:
            batch: 包含状态信息的字典
            actions_fixed: (bs, na, nt) - 固定的actions（one-hot编码）
        
        Returns:
            log_prob: (bs, 1) - 联合log概率
            entropy: (bs, 1) - 联合熵
        """
        # 复用forward的前半部分（特征提取和logits计算）
        entities = batch['entities']
        entity_mask = batch['entity_mask']
        task_mask = batch['task_mask']
        
        if th.isnan(entities).any():
            print(f"ERROR: NaN in input entities! Replacing with zeros.")
            entities = th.where(th.isnan(entities), th.zeros_like(entities), entities)
        
        na = self.args.n_agents
        nt = self.args.n_tasks
        bs = entities.shape[0]
        
        # 特征提取（与forward相同）
        entity2task = 1 - batch['entity2task_mask'].float()
        last_alloc = batch.get('last_alloc', th.zeros_like(entity2task[:, :na]))
        entity2task[:, :na] = last_alloc
        
        if not self.pi_pointer_net:
            entities = th.cat([entities, entity2task], dim=-1)
        
        x1 = self.fc1(entities)
        if th.isnan(x1).any():
            print(f"ERROR: NaN after fc1! Check network parameters.")
            x1 = th.where(th.isnan(x1), th.zeros_like(x1), x1)
        
        if self.pi_pointer_net:
            task_embed_out = self.task_embed(entity2task)
            if th.isnan(task_embed_out).any():
                print(f"ERROR: NaN after task_embed! Check network parameters.")
                task_embed_out = th.where(th.isnan(task_embed_out), th.zeros_like(task_embed_out), task_embed_out)
            x1 += task_embed_out
        
        # 非智能体实体处理
        nonagent_x1 = x1[:, na:]
        nonagent_mask = entity_mask[:, na:]
        if self.pi_pointer_net and self.subtask_mask:
            nonagent_attn_mask = groupmask2attnmask(batch['entity2task_mask'][:, na:])
        else:
            nonagent_attn_mask = groupmask2attnmask(nonagent_mask)
        nonagent_x2 = self.attn(F.relu(nonagent_x1), pre_mask=nonagent_attn_mask, post_mask=nonagent_mask)
        
        if self.pi_pointer_net:
            nonagent_entity2task = entity2task[:, na:]
            task_embeds = th.bmm(nonagent_entity2task.transpose(1, 2), nonagent_x2)
        else:
            task_embeds = nonagent_x2.mean(dim=1, keepdim=True).repeat(1, nt, 1)
        
        task_embeds = self.task_encoder(task_embeds)
        if th.isnan(task_embeds).any():
            print(f"ERROR: NaN after task_encoder! Check network parameters.")
            task_embeds = th.where(th.isnan(task_embeds), th.zeros_like(task_embeds), task_embeds)
        
        # 智能体特征处理
        agent_x1 = x1[:, :na]
        if self.pi_ag_attn:
            ag_mask = entity_mask[:, :na]
            active_mask = groupmask2attnmask(ag_mask)
            agent_attn_out = self.attn(F.relu(agent_x1), pre_mask=active_mask, post_mask=ag_mask)
            if th.isnan(agent_attn_out).any():
                print(f"ERROR: NaN after agent attention! Check network parameters.")
                agent_attn_out = th.where(th.isnan(agent_attn_out), th.zeros_like(agent_attn_out), agent_attn_out)
            agent_embeds = agent_x1 + agent_attn_out
        else:
            agent_embeds = agent_x1
        
        agent_embeds = self.agent_encoder(agent_embeds)
        if th.isnan(agent_embeds).any():
            print(f"ERROR: NaN after agent_encoder! Check network parameters.")
            agent_embeds = th.where(th.isnan(agent_embeds), th.zeros_like(agent_embeds), agent_embeds)
        
        # 使用公共函数计算logits，确保与forward一致
        masked_logits, agent_task_mask = self._compute_logits_parallel(agent_embeds, task_embeds, batch)
        
        # 计算固定actions的log_prob
        agent_mask = (1 - entity_mask[:, :na].float())  # (bs, na) - 1表示有效，0表示无效
        agent_mask_expanded = agent_mask  # (bs, na)
        
        # 检查每个agent是否有有效任务
        has_valid_task = (~agent_task_mask.bool()).any(dim=-1)  # (bs, na)
        if th.isnan(masked_logits).any():
            if not hasattr(self, '_nan_warning_count'):
                self._nan_warning_count = 0
            self._nan_warning_count += 1
            if self._nan_warning_count <= 1 or self._nan_warning_count % 100 == 0:
                print(f"Warning: NaN detected in masked_logits (count: {self._nan_warning_count}). Replacing.")
            NEG = -1e9
            masked_logits = th.where(
                agent_task_mask.bool(),
                th.tensor(NEG, device=masked_logits.device, dtype=masked_logits.dtype),
                th.where(th.isnan(masked_logits), th.zeros_like(masked_logits), masked_logits)
            )
        
        # 向量化计算log_prob和entropy（使用同一个dist，保证一致性）
        dist = Categorical(logits=masked_logits)  # (bs, na, nt) -> 自动广播为每个agent一个分布
        
        # 获取固定动作的索引
        action_indices = actions_fixed.argmax(dim=-1)  # (bs, na)
        
        # 向量化计算log_prob和entropy
        log_probs_per_agent = dist.log_prob(action_indices)  # (bs, na)
        entropy_per_agent = dist.entropy()  # (bs, na)
        
        # 对于无效agent或没有有效任务的agent，设为0
        valid_mask = agent_mask_expanded * has_valid_task  # (bs, na)
        log_probs_per_agent = log_probs_per_agent * valid_mask
        entropy_per_agent = entropy_per_agent * valid_mask
        
        # 检查-inf和NaN
        if th.isinf(log_probs_per_agent).any() or th.isnan(log_probs_per_agent).any():
            if not hasattr(self, '_log_prob_inf_nan_count'):
                self._log_prob_inf_nan_count = 0
            self._log_prob_inf_nan_count += 1
            if self._log_prob_inf_nan_count <= 1 or self._log_prob_inf_nan_count % 100 == 0:
                n_inf = th.isinf(log_probs_per_agent).sum().item()
                n_nan = th.isnan(log_probs_per_agent).sum().item()
                print(f"Warning: inf/NaN in log_probs_per_agent (inf: {n_inf}, NaN: {n_nan}, count: {self._log_prob_inf_nan_count}). Replacing with zeros.")
            log_probs_per_agent = th.where(
                th.isinf(log_probs_per_agent) | th.isnan(log_probs_per_agent),
                th.zeros_like(log_probs_per_agent),
                log_probs_per_agent
            )
        
        # 联合log概率（只对有效agent且有有效任务的求和）
        joint_log_prob = log_probs_per_agent.sum(dim=1, keepdim=True)  # (bs, 1)
        
        # 检查-inf和NaN
        if th.isinf(joint_log_prob).any() or th.isnan(joint_log_prob).any():
            if not hasattr(self, '_joint_log_prob_inf_nan_count'):
                self._joint_log_prob_inf_nan_count = 0
            self._joint_log_prob_inf_nan_count += 1
            if self._joint_log_prob_inf_nan_count <= 1 or self._joint_log_prob_inf_nan_count % 100 == 0:
                n_inf = th.isinf(joint_log_prob).sum().item()
                n_nan = th.isnan(joint_log_prob).sum().item()
                print(f"Warning: inf/NaN in joint_log_prob (inf: {n_inf}, NaN: {n_nan}, count: {self._joint_log_prob_inf_nan_count}). Replacing with zeros.")
            joint_log_prob = th.where(
                th.isinf(joint_log_prob) | th.isnan(joint_log_prob),
                th.zeros_like(joint_log_prob),
                joint_log_prob
            )
        
        # 计算熵（只对有效agent计算）
        if th.isinf(entropy_per_agent).any() or th.isnan(entropy_per_agent).any():
            entropy_per_agent = th.where(
                th.isinf(entropy_per_agent) | th.isnan(entropy_per_agent),
                th.zeros_like(entropy_per_agent),
                entropy_per_agent
            )
        
        joint_entropy = entropy_per_agent.sum(dim=1, keepdim=True)  # (bs, 1)
        
        return joint_log_prob, joint_entropy
    
    def _get_task_nonag_counts(self, batch):
        """
        获取非agent实体分配到任务的计数（类似原版ALMA）
        
        Args:
            batch: 包含状态信息的字典
        
        Returns:
            task_nonag_counts: (bs, nt) 或 None - 非agent实体分配到任务的计数
        """
        if not self.pi_pointer_net:
            return None
        
        entity2task_mask = batch['entity2task_mask']
        
        # 处理维度：确保entity2task_mask是3维 (bs, n_entities, nt)
        if entity2task_mask.dim() == 2:
            # (n_entities, nt) -> (1, n_entities, nt)
            entity2task_mask = entity2task_mask.unsqueeze(0)
        elif entity2task_mask.dim() == 3:
            # 已经是3维，检查是否需要处理batch size
            pass
        else:
            raise ValueError(f"Unexpected entity2task_mask dimension: {entity2task_mask.dim()}, expected 2 or 3")
        
        entity2task = 1 - entity2task_mask.float()  # (bs, n_entities, nt)
        na = self.args.n_agents
        
        # 确保有足够的维度进行切片
        if entity2task.shape[1] > na:
            nonagent_entity2task = entity2task[:, na:]  # (bs, n_nonagents, nt)
            task_nonag_counts = nonagent_entity2task.sum(dim=1) * COUNT_NORM_FACTOR  # (bs, nt)
        else:
            # 如果没有非agent实体，返回全零
            bs = entity2task.shape[0]
            nt = entity2task.shape[2] if entity2task.dim() >= 3 else self.args.n_tasks
            task_nonag_counts = th.zeros((bs, nt), device=entity2task.device, dtype=entity2task.dtype)
        
        return task_nonag_counts
    
    def _encode_features(self, batch):
        """
        提取agent和task的嵌入特征（用于AR生成）
        
        Args:
            batch: 包含状态信息的字典
        
        Returns:
            agent_embeds: (bs, na, embed_dim) - 智能体嵌入
            task_embeds: (bs, nt, embed_dim) - 任务嵌入
        """
        entities = batch['entities']
        entity_mask = batch['entity_mask']
        task_mask = batch['task_mask']
        
        # 处理维度：确保entities和entity_mask都是正确的维度
        # 从EliteBuffer采样时，应该已经是(bs, ne, ed)格式，但需要处理边界情况
        # 判断标准：如果dim==2且最后一个维度很大（>100），可能是(ne, ed)，需要添加batch维度
        # 如果dim==3，已经是(bs, ne, ed)，不需要处理
        if entities.dim() == 2:
            # 检查是否是(ne, ed)还是其他情况
            # 如果最后一个维度很大（比如>50），可能是特征维度，需要添加batch维度
            if entities.shape[-1] > 50:  # 假设特征维度>50
                entities = entities.unsqueeze(0)  # (1, ne, ed)
            # 否则可能是(bs, ne)的情况，不应该出现，但保持原样
        elif entities.dim() == 3:
            # 已经是(bs, ne, ed)，不需要处理
            pass
        else:
            raise ValueError(f"Unexpected entities dimension: {entities.dim()}, expected 2 or 3")
        
        if entity_mask.dim() == 1:
            # 如果是1维(ne,)，添加batch维度
            entity_mask = entity_mask.unsqueeze(0)  # (1, ne)
        elif entity_mask.dim() == 2:
            # 已经是(bs, ne)，不需要处理
            pass
        else:
            raise ValueError(f"Unexpected entity_mask dimension: {entity_mask.dim()}, expected 1 or 2")
        
        if task_mask.dim() == 1:
            # 如果是1维(nt,)，添加batch维度
            task_mask = task_mask.unsqueeze(0)  # (1, nt)
        elif task_mask.dim() == 2:
            # 已经是(bs, nt)，不需要处理
            pass
        else:
            raise ValueError(f"Unexpected task_mask dimension: {task_mask.dim()}, expected 1 or 2")
        
        if th.isnan(entities).any():
            print(f"ERROR: NaN in input entities! Replacing with zeros.")
            entities = th.where(th.isnan(entities), th.zeros_like(entities), entities)
        
        na = self.args.n_agents
        nt = self.args.n_tasks
        bs = entities.shape[0]  # 获取实际的batch size
        
        # 特征提取（与forward相同）
        entity2task_mask = batch['entity2task_mask']
        # 处理维度：确保entity2task_mask是3维
        if entity2task_mask.dim() == 2:
            entity2task_mask = entity2task_mask.unsqueeze(0)  # (1, ne, nt)
        elif entity2task_mask.dim() == 3:
            # 已经是(bs, ne, nt)，但需要确保batch size匹配
            if entity2task_mask.shape[0] != bs:
                if entity2task_mask.shape[0] == 1:
                    entity2task_mask = entity2task_mask.repeat(bs, 1, 1)
                else:
                    raise ValueError(f"entity2task_mask batch size {entity2task_mask.shape[0]} doesn't match entities batch size {bs}")
        
        entity2task = 1 - entity2task_mask.float()
        last_alloc = batch.get('last_alloc', None)
        if last_alloc is not None:
            # 处理维度：确保last_alloc是3维
            if last_alloc.dim() == 2:
                last_alloc = last_alloc.unsqueeze(0)  # (1, na, nt)
            entity2task[:, :na] = last_alloc
        else:
            entity2task[:, :na] = th.zeros_like(entity2task[:, :na])
        
        if not self.pi_pointer_net:
            entities = th.cat([entities, entity2task], dim=-1)
        
        x1 = self.fc1(entities)
        if th.isnan(x1).any():
            print(f"ERROR: NaN after fc1! Check network parameters.")
            x1 = th.where(th.isnan(x1), th.zeros_like(x1), x1)
        
        if self.pi_pointer_net:
            task_embed_out = self.task_embed(entity2task)
            if th.isnan(task_embed_out).any():
                print(f"ERROR: NaN after task_embed! Check network parameters.")
                task_embed_out = th.where(th.isnan(task_embed_out), th.zeros_like(task_embed_out), task_embed_out)
            x1 += task_embed_out
        
        # 非智能体实体处理
        nonagent_x1 = x1[:, na:]
        nonagent_mask = entity_mask[:, na:]
        if self.pi_pointer_net and self.subtask_mask:
            # 使用已经处理过维度的entity2task_mask（在函数开头已处理）
            nonagent_attn_mask = groupmask2attnmask(entity2task_mask[:, na:])
        else:
            nonagent_attn_mask = groupmask2attnmask(nonagent_mask)
        nonagent_x2 = self.attn(F.relu(nonagent_x1), pre_mask=nonagent_attn_mask, post_mask=nonagent_mask)
        
        if self.pi_pointer_net:
            nonagent_entity2task = entity2task[:, na:]
            task_embeds = th.bmm(nonagent_entity2task.transpose(1, 2), nonagent_x2)
        else:
            task_embeds = nonagent_x2.mean(dim=1, keepdim=True).repeat(1, nt, 1)
        
        task_embeds = self.task_encoder(task_embeds)
        if th.isnan(task_embeds).any():
            print(f"ERROR: NaN after task_encoder! Check network parameters.")
            task_embeds = th.where(th.isnan(task_embeds), th.zeros_like(task_embeds), task_embeds)
        
        # 智能体特征处理
        agent_x1 = x1[:, :na]
        if self.pi_ag_attn:
            ag_mask = entity_mask[:, :na]
            active_mask = groupmask2attnmask(ag_mask)
            agent_attn_out = self.attn(F.relu(agent_x1), pre_mask=active_mask, post_mask=ag_mask)
            if th.isnan(agent_attn_out).any():
                print(f"ERROR: NaN after agent attention! Check network parameters.")
                agent_attn_out = th.where(th.isnan(agent_attn_out), th.zeros_like(agent_attn_out), agent_attn_out)
            agent_embeds = agent_x1 + agent_attn_out
        else:
            agent_embeds = agent_x1
        
        agent_embeds = self.agent_encoder(agent_embeds)
        if th.isnan(agent_embeds).any():
            print(f"ERROR: NaN after agent_encoder! Check network parameters.")
            agent_embeds = th.where(th.isnan(agent_embeds), th.zeros_like(agent_embeds), agent_embeds)
        
        return agent_embeds, task_embeds
    
    def _compute_logits_for_agent_ar(self, agent_embed, task_embeds, task_ag_counts, task_nonag_counts, batch, agent_idx):
        """
        计算单个智能体在AR模式下的logits（原版ALMA方式：使用动态更新的task_embeds）
        
        Args:
            agent_embed: (bs, embed_dim) 或 (bs, 1, embed_dim) - 当前智能体的嵌入
            task_embeds: (bs, nt, embed_dim) - 任务嵌入（已动态更新，包含已分配agent信息）
            task_ag_counts: (bs, nt) - agent分配到任务的计数
            task_nonag_counts: (bs, nt) 或 None - 非agent实体分配到任务的计数
            batch: 包含状态信息的字典
            agent_idx: int - 智能体索引（用于mask）
        
        Returns:
            logits: (bs, nt) - 当前智能体对所有任务的logits
        """
        bs, nt, embed_dim = task_embeds.shape
        
        # 确保agent_embed是(b, 1, embed_dim)格式（用于指针网络）
        if agent_embed.dim() == 2:
            agent_embed = agent_embed.unsqueeze(1)  # (bs, 1, embed_dim)
        
        if self.pi_pointer_net:
            # 指针网络：使用点积注意力计算logits
            # 1. 编码计数信息
            if task_nonag_counts is not None:
                count_embeds = self.count_embed(th.stack([task_nonag_counts, task_ag_counts], dim=-1))  # (bs, nt, embed_dim)
            else:
                # 如果没有nonag_counts，只用ag_counts（填充0）
                task_nonag_counts_zero = th.zeros_like(task_ag_counts)
                count_embeds = self.count_embed(th.stack([task_nonag_counts_zero, task_ag_counts], dim=-1))  # (bs, nt, embed_dim)
            
            # 2. 将计数embedding加到task_embeds上
            task_embeds_with_count = task_embeds + count_embeds  # (bs, nt, embed_dim)
            
            # 3. 计算点积注意力（指针网络）
            logits = th.bmm(agent_embed, task_embeds_with_count.transpose(1, 2)).squeeze(1) / self.scale_factor  # (bs, nt)
        else:
            # 全连接层：为每个任务分别计算logit
            # 参考原版ALMA：cat([agent_embed, task_embeds], dim=1) -> out_fc -> logits
            agent_expanded = agent_embed.squeeze(1)  # (bs, embed_dim)
            # 为每个任务分别计算：需要将agent_embed扩展到每个任务
            agent_expanded = agent_expanded.unsqueeze(1).repeat(1, nt, 1)  # (bs, nt, embed_dim)
            combined = th.cat([agent_expanded, task_embeds], dim=-1)  # (bs, nt, embed_dim*2)
            # action_head输入是(bs, embed_dim*2)，输出是(bs, 1)
            # 需要为每个任务分别计算，所以需要reshape
            bs_nt, embed_dim_2 = combined.shape[0] * combined.shape[1], combined.shape[2]
            combined_flat = combined.view(bs_nt, embed_dim_2)  # (bs*nt, embed_dim*2)
            logits_flat = self.action_head(combined_flat).squeeze(-1)  # (bs*nt,)
            logits = logits_flat.view(bs, nt)  # (bs, nt)
        
        # 限制logits范围
        logits = th.clamp(logits, min=-50.0, max=50.0)
        
        # 应用任务掩码
        task_mask = batch['task_mask']  # (bs, nt)
        if task_mask.dim() == 2:
            task_mask_expanded = task_mask
        else:
            task_mask_expanded = task_mask
        
        NEG = -1e9
        masked_logits = logits.masked_fill(task_mask_expanded.bool(), NEG)
        
        # 应用avail_actions mask（如果存在）
        avail_actions = batch.get('avail_actions', None)
        if avail_actions is not None:
            if avail_actions.dim() == 3:
                agent_avail = avail_actions[:, agent_idx, :]  # (bs, nt_avail)
            else:
                agent_avail = avail_actions  # (bs, nt_avail)
            
            # 检查维度匹配：agent_avail的任务维度必须与masked_logits匹配
            if agent_avail.shape[-1] == masked_logits.shape[-1]:
                masked_logits = masked_logits.masked_fill((1 - agent_avail).bool(), NEG)
            else:
                # 维度不匹配，跳过avail_actions mask（使用task_mask即可）
                # 这种情况可能是avail_actions的任务维度与实际的n_tasks不一致
                pass
        
        return masked_logits
    
    def compute_allocation_autoreg(self, batch, test_mode=False):
        """
        自回归方式生成任务分配（用于rollout采样）
        
        Args:
            batch: 包含状态信息的字典
            test_mode: 是否为测试模式（True: greedy, False: sample）
        
        Returns:
            actions: (bs, na, nt) - 任务分配动作（one-hot编码）
            log_prob: (bs, 1) - 联合log概率
            entropy: (bs, 1) - 联合熵
            alloc_order: (bs, na) - 分配顺序（每个样本一个随机顺序）
        """
        # 1. 提取特征
        agent_embeds, task_embeds = self._encode_features(batch)
        
        bs, na, _ = agent_embeds.shape
        nt = self.args.n_tasks
        
        # 2. 初始化输出
        actions = th.zeros((bs, na, nt), device=agent_embeds.device, dtype=th.float32)
        joint_logp = th.zeros((bs, 1), device=agent_embeds.device)
        joint_ent = th.zeros((bs, 1), device=agent_embeds.device)
        
        # 3. 初始化状态：任务分配计数和任务embedding（会动态更新）
        task_ag_counts = th.zeros((bs, nt), device=agent_embeds.device, dtype=th.float32)
        # 获取task_nonag_counts（非agent实体分配到任务的计数）
        task_nonag_counts = self._get_task_nonag_counts(batch)  # (bs, nt) 或 None
        
        # 4. 随机顺序：每个样本一个随机permutation
        # 为每个样本生成一个随机顺序
        alloc_order = th.stack([
            th.randperm(na, device=agent_embeds.device) 
            for _ in range(bs)
        ], dim=0)  # (bs, na) - 每个样本一个随机顺序
        
        # 5. 自回归生成（原版ALMA方式：动态更新task_embeds）
        entity_mask = batch['entity_mask']
        agent_mask = entity_mask[:, :na]  # (bs, na)
        bs_idx = th.arange(bs, device=agent_embeds.device)  # (bs,)
        
        for step in range(na):
            # 5.1 获取当前要决策的智能体索引（每个样本可能不同）
            ai = alloc_order[:, step]  # (bs,) - 每个样本当前要决策的智能体id
            
            # 5.2 获取当前智能体的嵌入（需要gather，因为每个样本的ai可能不同）
            agent_embed = agent_embeds[bs_idx, ai, :]  # (bs, embed_dim)
            
            # 5.3 计算logits（向量化版本，避免逐个样本循环）
            # 使用批量操作，大幅提升性能
            agent_embed_expanded = agent_embed.unsqueeze(1)  # (bs, 1, embed_dim) - 用于指针网络
            
            if self.pi_pointer_net:
                # 指针网络：向量化计算
                # 1. 编码计数信息
                if task_nonag_counts is not None:
                    count_embeds = self.count_embed(th.stack([task_nonag_counts, task_ag_counts], dim=-1))  # (bs, nt, embed_dim)
                else:
                    task_nonag_counts_zero = th.zeros_like(task_ag_counts)
                    count_embeds = self.count_embed(th.stack([task_nonag_counts_zero, task_ag_counts], dim=-1))  # (bs, nt, embed_dim)
                
                # 2. 将计数embedding加到task_embeds上
                task_embeds_with_count = task_embeds + count_embeds  # (bs, nt, embed_dim)
                
                # 3. 批量计算点积注意力（指针网络）
                logits = th.bmm(agent_embed_expanded, task_embeds_with_count.transpose(1, 2)).squeeze(1) / self.scale_factor  # (bs, nt)
            else:
                # 全连接层：向量化计算
                agent_expanded = agent_embed.unsqueeze(1).repeat(1, nt, 1)  # (bs, nt, embed_dim)
                combined = th.cat([agent_expanded, task_embeds], dim=-1)  # (bs, nt, embed_dim*2)
                bs_nt, embed_dim_2 = combined.shape[0] * combined.shape[1], combined.shape[2]
                combined_flat = combined.view(bs_nt, embed_dim_2)  # (bs*nt, embed_dim*2)
                logits_flat = self.action_head(combined_flat).squeeze(-1)  # (bs*nt,)
                logits = logits_flat.view(bs, nt)  # (bs, nt)
            
            # 限制logits范围
            logits = th.clamp(logits, min=-50.0, max=50.0)
            
            # 应用任务掩码（向量化）
            task_mask = batch['task_mask']  # (bs, nt)
            if task_mask.dim() == 2:
                task_mask_expanded = task_mask
            else:
                task_mask_expanded = task_mask
            NEG = -1e9
            logits = logits.masked_fill(task_mask_expanded.bool(), NEG)
            
            # 应用avail_actions mask（向量化）
            avail_actions = batch.get('avail_actions', None)
            if avail_actions is not None:
                if avail_actions.dim() == 3:
                    # avail_actions: (bs, na, nt_avail)，需要gather每个样本对应的agent
                    agent_avail = avail_actions[bs_idx, ai, :]  # (bs, nt_avail)
                else:
                    agent_avail = avail_actions  # (bs, nt_avail)
                
                # 检查维度匹配：agent_avail的任务维度必须与logits匹配
                if agent_avail.shape[-1] == logits.shape[-1]:
                    logits = logits.masked_fill((1 - agent_avail).bool(), NEG)
                # 如果维度不匹配，跳过avail_actions mask（使用task_mask即可）
            
            # 5.4 创建分布并采样
            dist = Categorical(logits=logits)
            
            if test_mode:
                a_idx = dist.probs.argmax(dim=-1)  # (bs,)
            else:
                a_idx = dist.sample()  # (bs,)
            
            # 5.5 转换为one-hot并保存（向量化）
            hard_ac = F.one_hot(a_idx, num_classes=nt).float()  # (bs, nt)
            # 向量化索引：actions[bs_idx, ai, :] = hard_ac
            actions[bs_idx, ai, :] = hard_ac
            
            # 5.6 计算log_prob和entropy（只对有效agent）
            curr_agent_valid = (1 - agent_mask[bs_idx, ai]).float()  # (bs,)
            if curr_agent_valid.any():
                logp = dist.log_prob(a_idx).unsqueeze(-1)  # (bs, 1)
                ent = dist.entropy().unsqueeze(-1)  # (bs, 1)
                joint_logp += logp * curr_agent_valid.unsqueeze(-1)
                joint_ent += ent * curr_agent_valid.unsqueeze(-1)
            
            # 5.7 更新状态（原版ALMA方式：动态更新task_embeds和task_ag_counts）
            valid_mask = curr_agent_valid.bool()  # (bs,)
            if valid_mask.any():
                # 更新task_ag_counts
                task_ag_counts[valid_mask].scatter_add_(
                    dim=1,
                    index=a_idx[valid_mask].unsqueeze(-1),
                    src=th.ones((valid_mask.sum(), 1), device=task_ag_counts.device) * COUNT_NORM_FACTOR
                )
                
                # 动态更新task_embeds（原版ALMA的核心机制）
                # 向量化版本，避免逐个样本循环，大幅提升性能
                curr_agent_valid_expanded = curr_agent_valid.view(bs, 1, 1)  # (bs, 1, 1) - 用于广播
                hard_ac_expanded = hard_ac.unsqueeze(-1)  # (bs, nt, 1)
                
                # 获取当前agent的embedding（批量）
                agent_embed_for_update = agent_embed.unsqueeze(1)  # (bs, 1, embed_dim)
                
                if self.pi_pointer_net:
                    if self.sel_task_upd:
                        # 只更新被选中的任务embedding（向量化）
                        embed_upd_in = th.cat([task_embeds, agent_embed_for_update.expand(-1, nt, -1)], dim=2)  # (bs, nt, embed_dim*2)
                        task_embeds_update = self.embed_update(F.relu(embed_upd_in))  # (bs, nt, embed_dim)
                        # 使用tensor mask，保持梯度链，向量化更新
                        update_mask = hard_ac_expanded * curr_agent_valid_expanded  # (bs, nt, 1)
                        task_embeds = task_embeds + task_embeds_update * update_mask  # 函数式更新，保持梯度链
                    else:
                        # 更新所有任务embedding（向量化）
                        sel_task_embed = (task_embeds * hard_ac_expanded).sum(dim=1, keepdim=True)  # (bs, 1, embed_dim)
                        embed_upd_in = th.cat([task_embeds, agent_embed_for_update.expand(-1, nt, -1), sel_task_embed.expand(-1, nt, -1)], dim=2)  # (bs, nt, embed_dim*3)
                        task_embeds_update = self.embed_update(F.relu(embed_upd_in))  # (bs, nt, embed_dim)
                        # 使用tensor mask，保持梯度链，向量化更新
                        task_embeds = task_embeds + task_embeds_update * curr_agent_valid_expanded  # 函数式更新，保持梯度链
                else:
                    # 不使用指针网络的情况（需要逐个样本处理，因为每个样本选中的任务不同）
                    # 但可以优化：只对有效样本处理
                    valid_indices = th.where(valid_mask)[0]
                    for b_idx in valid_indices:
                        b = b_idx.item()  # 只用于索引
                        task_j = a_idx[b].item()  # 只用于索引
                        hard_ac_b_full = hard_ac[b:b+1, :]  # (1, nt)
                        embed_upd_in = th.cat([
                            task_embeds[b:b+1, task_j:task_j+1, :].squeeze(1),  # (1, embed_dim)
                            agent_embed_for_update[b:b+1].squeeze(1),  # (1, embed_dim)
                            hard_ac_b_full  # (1, nt)
                        ], dim=1)  # (1, embed_dim + embed_dim + nt)
                        task_embeds_update = self.embed_update(F.relu(embed_upd_in))  # (1, embed_dim)
                        # 使用tensor mask，保持梯度链
                        task_embeds = task_embeds.clone()  # 避免in-place
                        task_embeds[b:b+1, task_j:task_j+1, :] = task_embeds[b:b+1, task_j:task_j+1, :] + task_embeds_update.unsqueeze(1) * curr_agent_valid_expanded[b:b+1]
        
        # 5.8 清零无效agent的动作（关键修复）
        # 确保无效agent的动作为0，与并行模式保持一致
        agent_mask_final = (1 - agent_mask).unsqueeze(-1)  # (bs, na, 1) - 1表示有效，0表示无效
        actions = actions * agent_mask_final  # (bs, na, nt) - 无效agent的动作清零
        
        return actions, joint_logp, joint_ent, alloc_order
    
    def compute_logprob_for_actions_autoreg(self, batch, actions_fixed, alloc_order=None):
        """
        自回归方式重算固定actions的log_prob（用于PPO训练）
        
        Args:
            batch: 包含状态信息的字典
            actions_fixed: (bs, na, nt) - 固定的actions（one-hot编码）
            alloc_order: (na,) 或 (bs, na) - 分配顺序，如果为None则使用固定顺序
        
        Returns:
            joint_log_prob: (bs, 1) - 联合log概率
            joint_entropy: (bs, 1) - 联合熵
        """
        # 1. 提取特征
        agent_embeds, task_embeds = self._encode_features(batch)
        
        bs, na, _ = agent_embeds.shape
        nt = self.args.n_tasks
        
        # 1.1 规范化batch中的所有tensor，确保维度匹配
        # 处理task_mask - 确保是(bs, nt)
        task_mask = batch.get('task_mask', None)
        if task_mask is not None:
            if task_mask.dim() == 1:
                # (nt,) -> (bs, nt) 或 (bs,) -> 需要判断
                if task_mask.shape[0] == nt:
                    task_mask = task_mask.unsqueeze(0).repeat(bs, 1)  # (bs, nt)
                elif task_mask.shape[0] == bs:
                    # 可能是错误的维度，假设是(nt,)被误认为是(bs,)
                    if bs == 1:
                        task_mask = task_mask.unsqueeze(0)  # (1, nt) 如果nt==bs
                    else:
                        # 尝试reshape
                        if task_mask.shape[0] % nt == 0:
                            task_mask = task_mask.view(-1, nt)
                            if task_mask.shape[0] != bs:
                                if task_mask.shape[0] > bs:
                                    task_mask = task_mask[:bs, :]
                                else:
                                    padding = th.zeros(bs - task_mask.shape[0], nt, device=task_mask.device, dtype=task_mask.dtype)
                                    task_mask = th.cat([task_mask, padding], dim=0)
                        else:
                            # 填充或截断到nt，然后repeat
                            if task_mask.shape[0] > nt:
                                task_mask = task_mask[:nt].unsqueeze(0).repeat(bs, 1)
                            else:
                                padding = th.zeros(bs, nt - task_mask.shape[0], device=task_mask.device, dtype=task_mask.dtype)
                                task_mask = th.cat([task_mask.unsqueeze(0).repeat(bs, 1), padding], dim=1)
                else:
                    # 长度不匹配，尝试reshape
                    if task_mask.shape[0] % nt == 0:
                        task_mask = task_mask.view(-1, nt)
                        if task_mask.shape[0] != bs:
                            if task_mask.shape[0] > bs:
                                task_mask = task_mask[:bs, :]
                            else:
                                padding = th.zeros(bs - task_mask.shape[0], nt, device=task_mask.device, dtype=task_mask.dtype)
                                task_mask = th.cat([task_mask, padding], dim=0)
                    else:
                        # 无法reshape，创建新的
                        task_mask = th.zeros(bs, nt, device=task_mask.device, dtype=task_mask.dtype)
            elif task_mask.dim() == 2:
                # 已经是2维，检查形状
                if task_mask.shape == (bs, nt):
                    # 形状正确，直接使用
                    pass
                elif task_mask.shape[0] == bs and task_mask.shape[1] != nt:
                    # batch size正确但nt不匹配
                    if task_mask.shape[1] > nt:
                        task_mask = task_mask[:, :nt]
                    else:
                        padding = th.zeros(bs, nt - task_mask.shape[1], device=task_mask.device, dtype=task_mask.dtype)
                        task_mask = th.cat([task_mask, padding], dim=1)
                elif task_mask.shape[0] != bs and task_mask.shape[1] == nt:
                    # nt正确但bs不匹配
                    if task_mask.shape[0] == 1:
                        task_mask = task_mask.repeat(bs, 1)
                    elif task_mask.shape[0] > bs:
                        task_mask = task_mask[:bs, :]
                    else:
                        repeat_times = bs // task_mask.shape[0]
                        remainder = bs % task_mask.shape[0]
                        if remainder == 0:
                            task_mask = task_mask.repeat(repeat_times, 1)
                        else:
                            task_mask = th.cat([
                                task_mask.repeat(repeat_times, 1),
                                task_mask[:remainder]
                            ], dim=0)
                else:
                    # 完全重新创建匹配的task_mask
                    task_mask = th.zeros(bs, nt, device=task_mask.device, dtype=task_mask.dtype)
            else:
                # 其他维度，创建新的
                task_mask = th.zeros(bs, nt, device=agent_embeds.device)
            batch['task_mask'] = task_mask
        else:
            # 如果没有task_mask，创建全零的（表示所有任务都有效）
            batch['task_mask'] = th.zeros(bs, nt, device=agent_embeds.device)
        
        # 2. 处理alloc_order
        if alloc_order is None:
            # 固定顺序：0..na-1
            alloc_order = th.arange(na, device=agent_embeds.device).unsqueeze(0).repeat(bs, 1)  # (bs, na)
        elif alloc_order.dim() == 1:
            # 如果是一维，检查长度
            if alloc_order.shape[0] == na:
                # 长度正确，扩展到batch维度
                alloc_order = alloc_order.unsqueeze(0).repeat(bs, 1)  # (bs, na)
            elif alloc_order.shape[0] == bs:
                # 如果长度等于bs，可能是错误的维度，尝试reshape
                # 这种情况不应该出现，但处理一下
                if bs == 1:
                    # 如果bs=1，可能是 (na,) 被误认为是 (bs,)
                    alloc_order = alloc_order[:na].unsqueeze(0)  # (1, na)
                else:
                    raise ValueError(f"Unexpected alloc_order shape: {alloc_order.shape}, expected (na,) or (bs, na) with na={na}, bs={bs}")
            else:
                # 长度不匹配，截断或报错
                if alloc_order.shape[0] > na:
                    alloc_order = alloc_order[:na].unsqueeze(0).repeat(bs, 1)  # (bs, na)
                else:
                    raise ValueError(f"alloc_order length {alloc_order.shape[0]} doesn't match na={na}")
        elif alloc_order.dim() == 2:
            # 已经是2维，检查形状
            if alloc_order.shape == (bs, na):
                # 形状正确，直接使用
                pass
            elif alloc_order.shape[0] == bs and alloc_order.shape[1] != na:
                # batch size正确但na不匹配，截断或填充
                if alloc_order.shape[1] > na:
                    alloc_order = alloc_order[:, :na]  # (bs, na)
                else:
                    # 填充
                    padding = th.zeros(bs, na - alloc_order.shape[1], device=alloc_order.device, dtype=alloc_order.dtype)
                    alloc_order = th.cat([alloc_order, padding], dim=1)  # (bs, na)
            elif alloc_order.shape[0] != bs and alloc_order.shape[1] == na:
                # na正确但bs不匹配
                if alloc_order.shape[0] == 1:
                    # (1, na) -> (bs, na)
                    alloc_order = alloc_order.repeat(bs, 1)
                elif alloc_order.shape[0] > bs:
                    # alloc_order的batch size更大，取前bs个
                    alloc_order = alloc_order[:bs, :]  # (bs, na)
                else:
                    # alloc_order的batch size更小，repeat到bs
                    repeat_times = bs // alloc_order.shape[0]
                    remainder = bs % alloc_order.shape[0]
                    if remainder == 0:
                        alloc_order = alloc_order.repeat(repeat_times, 1)  # (bs, na)
                    else:
                        alloc_order = th.cat([
                            alloc_order.repeat(repeat_times, 1),
                            alloc_order[:remainder]
                        ], dim=0)  # (bs, na)
            else:
                raise ValueError(f"Unexpected alloc_order shape: {alloc_order.shape}, expected (bs, na) with bs={bs}, na={na}")
        else:
            raise ValueError(f"Unexpected alloc_order dimension: {alloc_order.dim()}, expected 1 or 2")
        
        # 3. 初始化输出
        joint_logp = th.zeros((bs, 1), device=agent_embeds.device)
        joint_ent = th.zeros((bs, 1), device=agent_embeds.device)
        
        # 4. 初始化状态：任务分配计数和任务embedding（会动态更新）
        task_ag_counts = th.zeros((bs, nt), device=agent_embeds.device, dtype=th.float32)
        task_nonag_counts = self._get_task_nonag_counts(batch)  # (bs, nt) 或 None
        
        # 规范化task_nonag_counts的维度，确保batch size匹配
        if task_nonag_counts is not None:
            if task_nonag_counts.dim() == 1:
                # 如果是1维，检查长度
                if task_nonag_counts.shape[0] == nt:
                    # (nt,) -> (bs, nt)
                    task_nonag_counts = task_nonag_counts.unsqueeze(0).repeat(bs, 1)
                elif task_nonag_counts.shape[0] == bs:
                    # 如果长度等于bs，可能是错误的维度，需要reshape
                    # 这种情况不应该出现，但处理一下
                    if bs == 1:
                        task_nonag_counts = task_nonag_counts.unsqueeze(0)  # (1, nt)
                    else:
                        # 尝试reshape为(bs, nt)，但需要知道nt
                        # 这里假设可以reshape
                        task_nonag_counts = task_nonag_counts.view(bs, -1)
                        if task_nonag_counts.shape[1] != nt:
                            # 如果reshape后nt不匹配，截断或填充
                            if task_nonag_counts.shape[1] > nt:
                                task_nonag_counts = task_nonag_counts[:, :nt]
                            else:
                                padding = th.zeros(bs, nt - task_nonag_counts.shape[1], device=task_nonag_counts.device, dtype=task_nonag_counts.dtype)
                                task_nonag_counts = th.cat([task_nonag_counts, padding], dim=1)
                else:
                    # 长度不匹配，尝试reshape或报错
                    if task_nonag_counts.shape[0] % nt == 0:
                        # 可能是展平的(bs*nt,)，reshape为(bs, nt)
                        task_nonag_counts = task_nonag_counts.view(-1, nt)
                        if task_nonag_counts.shape[0] != bs:
                            if task_nonag_counts.shape[0] > bs:
                                task_nonag_counts = task_nonag_counts[:bs, :]
                            else:
                                padding = th.zeros(bs - task_nonag_counts.shape[0], nt, device=task_nonag_counts.device, dtype=task_nonag_counts.dtype)
                                task_nonag_counts = th.cat([task_nonag_counts, padding], dim=0)
                    else:
                        raise ValueError(f"task_nonag_counts shape {task_nonag_counts.shape} cannot be reshaped to (bs, nt) with bs={bs}, nt={nt}")
            elif task_nonag_counts.dim() == 2:
                # 已经是2维，检查形状
                if task_nonag_counts.shape == (bs, nt):
                    # 形状正确，直接使用
                    pass
                elif task_nonag_counts.shape[0] == bs and task_nonag_counts.shape[1] != nt:
                    # batch size正确但nt不匹配，截断或填充
                    if task_nonag_counts.shape[1] > nt:
                        task_nonag_counts = task_nonag_counts[:, :nt]
                    else:
                        padding = th.zeros(bs, nt - task_nonag_counts.shape[1], device=task_nonag_counts.device, dtype=task_nonag_counts.dtype)
                        task_nonag_counts = th.cat([task_nonag_counts, padding], dim=1)
                elif task_nonag_counts.shape[0] != bs and task_nonag_counts.shape[1] == nt:
                    # nt正确但bs不匹配
                    if task_nonag_counts.shape[0] == 1:
                        # (1, nt) -> (bs, nt)
                        task_nonag_counts = task_nonag_counts.repeat(bs, 1)
                    elif task_nonag_counts.shape[0] > bs:
                        # 取前bs个
                        task_nonag_counts = task_nonag_counts[:bs, :]
                    else:
                        # repeat到bs
                        repeat_times = bs // task_nonag_counts.shape[0]
                        remainder = bs % task_nonag_counts.shape[0]
                        if remainder == 0:
                            task_nonag_counts = task_nonag_counts.repeat(repeat_times, 1)
                        else:
                            task_nonag_counts = th.cat([
                                task_nonag_counts.repeat(repeat_times, 1),
                                task_nonag_counts[:remainder]
                            ], dim=0)
                else:
                    raise ValueError(f"task_nonag_counts shape {task_nonag_counts.shape} doesn't match (bs, nt) with bs={bs}, nt={nt}")
            else:
                raise ValueError(f"Unexpected task_nonag_counts dimension: {task_nonag_counts.dim()}, expected 1 or 2")
        
        # 5. 自回归重算（原版ALMA方式：动态更新task_embeds）
        entity_mask = batch['entity_mask']
        # 处理entity_mask维度：如果是1维，需要扩展为2维
        if entity_mask.dim() == 1:
            # 如果是1维 (n_entities,)，扩展为 (1, n_entities) 然后repeat到batch size
            entity_mask = entity_mask.unsqueeze(0).repeat(bs, 1)  # (bs, n_entities)
        agent_mask = entity_mask[:, :na]  # (bs, na)
        bs_idx = th.arange(bs, device=agent_embeds.device)  # (bs,)
        
        for step in range(na):
            # 5.1 获取当前要决策的智能体索引（每个样本可能不同）
            ai = alloc_order[:, step]  # (bs,) - 每个样本当前要决策的智能体id
            
            # 5.2 获取当前智能体的嵌入
            agent_embed = agent_embeds[bs_idx, ai, :]  # (bs, embed_dim)
            
            # 5.3 计算logits（向量化版本，避免逐个样本循环）
            # 使用批量操作，大幅提升性能
            agent_embed_expanded = agent_embed.unsqueeze(1)  # (bs, 1, embed_dim) - 用于指针网络
            
            if self.pi_pointer_net:
                # 指针网络：向量化计算
                # 1. 编码计数信息
                if task_nonag_counts is not None:
                    # 确保两个tensor的形状完全匹配
                    if task_nonag_counts.shape != task_ag_counts.shape:
                        # 如果形状不匹配，调整task_nonag_counts以匹配task_ag_counts
                        if task_nonag_counts.shape[0] != task_ag_counts.shape[0]:
                            # batch size不匹配
                            if task_nonag_counts.shape[0] == 1:
                                task_nonag_counts = task_nonag_counts.repeat(task_ag_counts.shape[0], 1)
                            elif task_nonag_counts.shape[0] > task_ag_counts.shape[0]:
                                task_nonag_counts = task_nonag_counts[:task_ag_counts.shape[0], :]
                            else:
                                repeat_times = task_ag_counts.shape[0] // task_nonag_counts.shape[0]
                                remainder = task_ag_counts.shape[0] % task_nonag_counts.shape[0]
                                if remainder == 0:
                                    task_nonag_counts = task_nonag_counts.repeat(repeat_times, 1)
                                else:
                                    task_nonag_counts = th.cat([
                                        task_nonag_counts.repeat(repeat_times, 1),
                                        task_nonag_counts[:remainder]
                                    ], dim=0)
                        if task_nonag_counts.shape[1] != task_ag_counts.shape[1]:
                            # nt不匹配
                            if task_nonag_counts.shape[1] > task_ag_counts.shape[1]:
                                task_nonag_counts = task_nonag_counts[:, :task_ag_counts.shape[1]]
                            else:
                                padding = th.zeros(task_nonag_counts.shape[0], task_ag_counts.shape[1] - task_nonag_counts.shape[1], 
                                                 device=task_nonag_counts.device, dtype=task_nonag_counts.dtype)
                                task_nonag_counts = th.cat([task_nonag_counts, padding], dim=1)
                    count_embeds = self.count_embed(th.stack([task_nonag_counts, task_ag_counts], dim=-1))  # (bs, nt, embed_dim)
                else:
                    task_nonag_counts_zero = th.zeros_like(task_ag_counts)
                    count_embeds = self.count_embed(th.stack([task_nonag_counts_zero, task_ag_counts], dim=-1))  # (bs, nt, embed_dim)
                
                # 2. 将计数embedding加到task_embeds上
                task_embeds_with_count = task_embeds + count_embeds  # (bs, nt, embed_dim)
                
                # 3. 批量计算点积注意力（指针网络）
                logits = th.bmm(agent_embed_expanded, task_embeds_with_count.transpose(1, 2)).squeeze(1) / self.scale_factor  # (bs, nt)
            else:
                # 全连接层：向量化计算
                agent_expanded = agent_embed.unsqueeze(1).repeat(1, nt, 1)  # (bs, nt, embed_dim)
                combined = th.cat([agent_expanded, task_embeds], dim=-1)  # (bs, nt, embed_dim*2)
                bs_nt, embed_dim_2 = combined.shape[0] * combined.shape[1], combined.shape[2]
                combined_flat = combined.view(bs_nt, embed_dim_2)  # (bs*nt, embed_dim*2)
                logits_flat = self.action_head(combined_flat).squeeze(-1)  # (bs*nt,)
                logits = logits_flat.view(bs, nt)  # (bs, nt)
            
            # 限制logits范围
            logits = th.clamp(logits, min=-50.0, max=50.0)
            
            # 应用任务掩码（向量化）
            # task_mask已经在方法开始时规范化过了，确保是(bs, nt)
            task_mask = batch['task_mask']  # (bs, nt)
            # 确保task_mask和logits的形状完全匹配
            if task_mask.shape != logits.shape:
                # 如果形状不匹配，调整task_mask以匹配logits
                if task_mask.shape[0] == logits.shape[0] and task_mask.shape[1] != logits.shape[1]:
                    # batch size匹配，但nt不匹配
                    if task_mask.shape[1] > logits.shape[1]:
                        task_mask = task_mask[:, :logits.shape[1]]
                    else:
                        padding = th.zeros(task_mask.shape[0], logits.shape[1] - task_mask.shape[1], 
                                         device=task_mask.device, dtype=task_mask.dtype)
                        task_mask = th.cat([task_mask, padding], dim=1)
                elif task_mask.shape[0] != logits.shape[0] and task_mask.shape[1] == logits.shape[1]:
                    # nt匹配，但batch size不匹配
                    if task_mask.shape[0] == 1:
                        task_mask = task_mask.repeat(logits.shape[0], 1)
                    elif task_mask.shape[0] > logits.shape[0]:
                        task_mask = task_mask[:logits.shape[0], :]
                    else:
                        repeat_times = logits.shape[0] // task_mask.shape[0]
                        remainder = logits.shape[0] % task_mask.shape[0]
                        if remainder == 0:
                            task_mask = task_mask.repeat(repeat_times, 1)
                        else:
                            task_mask = th.cat([
                                task_mask.repeat(repeat_times, 1),
                                task_mask[:remainder]
                            ], dim=0)
                else:
                    # 完全重新创建匹配的task_mask
                    task_mask = th.zeros_like(logits, dtype=task_mask.dtype)
            NEG = -1e9
            logits = logits.masked_fill(task_mask.bool(), NEG)
            
            # 应用avail_actions mask（向量化）
            avail_actions = batch.get('avail_actions', None)
            if avail_actions is not None:
                if avail_actions.dim() == 3:
                    # avail_actions: (bs, na, nt_avail)，需要gather每个样本对应的agent
                    agent_avail = avail_actions[bs_idx, ai, :]  # (bs, nt_avail)
                else:
                    agent_avail = avail_actions  # (bs, nt_avail)
                
                # 检查维度匹配：agent_avail的任务维度必须与logits匹配
                if agent_avail.shape[-1] == logits.shape[-1]:
                    logits = logits.masked_fill((1 - agent_avail).bool(), NEG)
                # 如果维度不匹配，跳过avail_actions mask（使用task_mask即可）
            
            # 5.4 创建分布
            dist = Categorical(logits=logits)
            
            # 5.5 获取固定动作的索引
            a_idx = actions_fixed[bs_idx, ai, :].argmax(dim=-1)  # (bs,)
            hard_ac = F.one_hot(a_idx, num_classes=nt).float()  # (bs, nt)
            
            # 5.6 计算log_prob和entropy（只对有效agent）
            curr_agent_valid = (1 - agent_mask[bs_idx, ai]).float()  # (bs,)
            if curr_agent_valid.any():
                logp = dist.log_prob(a_idx).unsqueeze(-1)  # (bs, 1)
                ent = dist.entropy().unsqueeze(-1)  # (bs, 1)
                joint_logp += logp * curr_agent_valid.unsqueeze(-1)
                joint_ent += ent * curr_agent_valid.unsqueeze(-1)
            
            # 5.7 更新状态（原版ALMA方式：动态更新task_embeds和task_ag_counts）
            valid_mask = curr_agent_valid.bool()  # (bs,)
            if valid_mask.any():
                # 更新task_ag_counts
                task_ag_counts[valid_mask].scatter_add_(
                    dim=1,
                    index=a_idx[valid_mask].unsqueeze(-1),
                    src=th.ones((valid_mask.sum(), 1), device=task_ag_counts.device) * COUNT_NORM_FACTOR
                )
                
                # 动态更新task_embeds
                # 向量化版本，避免逐个样本循环，大幅提升性能
                curr_agent_valid_expanded = curr_agent_valid.view(bs, 1, 1)  # (bs, 1, 1) - 用于广播
                hard_ac_expanded = hard_ac.unsqueeze(-1)  # (bs, nt, 1)
                
                # 获取当前agent的embedding（批量）
                agent_embed_for_update = agent_embed.unsqueeze(1)  # (bs, 1, embed_dim)
                
                if self.pi_pointer_net:
                    if self.sel_task_upd:
                        # 只更新被选中的任务embedding（向量化）
                        embed_upd_in = th.cat([task_embeds, agent_embed_for_update.expand(-1, nt, -1)], dim=2)  # (bs, nt, embed_dim*2)
                        task_embeds_update = self.embed_update(F.relu(embed_upd_in))  # (bs, nt, embed_dim)
                        # 使用tensor mask，保持梯度链，向量化更新
                        update_mask = hard_ac_expanded * curr_agent_valid_expanded  # (bs, nt, 1)
                        task_embeds = task_embeds + task_embeds_update * update_mask  # 函数式更新，保持梯度链
                    else:
                        # 更新所有任务embedding（向量化）
                        sel_task_embed = (task_embeds * hard_ac_expanded).sum(dim=1, keepdim=True)  # (bs, 1, embed_dim)
                        embed_upd_in = th.cat([task_embeds, agent_embed_for_update.expand(-1, nt, -1), sel_task_embed.expand(-1, nt, -1)], dim=2)  # (bs, nt, embed_dim*3)
                        task_embeds_update = self.embed_update(F.relu(embed_upd_in))  # (bs, nt, embed_dim)
                        # 使用tensor mask，保持梯度链，向量化更新
                        task_embeds = task_embeds + task_embeds_update * curr_agent_valid_expanded  # 函数式更新，保持梯度链
                else:
                    # 不使用指针网络的情况（需要逐个样本处理，因为每个样本选中的任务不同）
                    # 但可以优化：只对有效样本处理
                    valid_indices = th.where(valid_mask)[0]
                    for b_idx in valid_indices:
                        b = b_idx.item()  # 只用于索引
                        task_j = a_idx[b].item()  # 只用于索引
                        hard_ac_b_full = hard_ac[b:b+1, :]  # (1, nt)
                        embed_upd_in = th.cat([
                            task_embeds[b:b+1, task_j:task_j+1, :].squeeze(1),  # (1, embed_dim)
                            agent_embed_for_update[b:b+1].squeeze(1),  # (1, embed_dim)
                            hard_ac_b_full  # (1, nt)
                        ], dim=1)  # (1, embed_dim + embed_dim + nt)
                        task_embeds_update = self.embed_update(F.relu(embed_upd_in))  # (1, embed_dim)
                        # 使用tensor mask，保持梯度链
                        task_embeds = task_embeds.clone()  # 避免in-place
                        task_embeds[b:b+1, task_j:task_j+1, :] = task_embeds[b:b+1, task_j:task_j+1, :] + task_embeds_update.unsqueeze(1) * curr_agent_valid_expanded[b:b+1]
        
        return joint_logp, joint_ent