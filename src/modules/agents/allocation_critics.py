import torch as th
import torch.nn as nn
import torch.nn.functional as F
from ..layers import EntityAttentionLayer
from .allocation_common import groupmask2attnmask, TaskEmbedder, CountEmbedder
from utils.rl_utils import ExponentialMeanStd


class StandardAllocCritic(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args

        self.args = args
        self.in_fc_ent = nn.Linear(input_shape, args.alloc_embed_dim)
        self.in_fc_alloc = TaskEmbedder(args.alloc_embed_dim, args)
        self.attn = EntityAttentionLayer(args.alloc_embed_dim,
                                         args.alloc_embed_dim,
                                         args.alloc_embed_dim, args,
                                         n_heads=args.alloc_n_heads,
                                         use_layernorm=False)
        self.count_embed = CountEmbedder(args.alloc_embed_dim, args)
        self.out_dim = 1
        self.out_fc = nn.Linear(args.alloc_embed_dim, self.out_dim)

        self.use_popart = self.args.popart
        if self.use_popart:
            self.targ_rms = ExponentialMeanStd(alpha=0.01)
            self.popart_weight = nn.parameter.Parameter(
                th.ones(1, self.out_dim))
            self.popart_bias = nn.parameter.Parameter(
                th.zeros(1, self.out_dim))

    def load_state_dict(self, state_dict):
        if self.use_popart:
            targ_rms_state_dict, state_dict = state_dict
            self.targ_rms.load_state_dict(targ_rms_state_dict)
        super().load_state_dict(state_dict)

    def state_dict(self):
        if self.use_popart:
            return self.targ_rms.state_dict(), super().state_dict()
        return super().state_dict()

    def popart_update(self, targets, mask):
        assert self.use_popart
        if self.targ_rms.mean is not None:
            old_mean = self.targ_rms.mean.clone()
            old_var = self.targ_rms.var.clone()
            self.targ_rms.update(targets, mask)
        else:
            self.targ_rms.update(targets, mask)
            old_mean = self.targ_rms.mean.clone()
            old_var = self.targ_rms.var.clone()
        sd_ratio = old_var.sqrt() / self.targ_rms.var.sqrt()
        self.popart_weight.data.mul_(sd_ratio)
        self.popart_bias.data.mul_(sd_ratio).add_((old_mean - self.targ_rms.mean) / self.targ_rms.var.sqrt())
        return (targets - self.targ_rms.mean) / self.targ_rms.var.sqrt()

    def denormalize(self, q):
        if self.targ_rms.mean is None or not self.use_popart:
            return q
        return q * self.targ_rms.var.sqrt() + self.targ_rms.mean

    def forward(self, batch, override_alloc=None, test_mode=False, calc_stats=False):
        entities = batch['entities']
        bs = entities.shape[0]
        entity_mask = batch['entity_mask']
        attn_mask = groupmask2attnmask(entity_mask)
        entity2task = 1 - batch['entity2task_mask'].float()
        multi_eval = False
        repeat_fn = lambda x: x
        if override_alloc is not None:
            if len(override_alloc.shape) == 4:
                multi_eval = True
                bs, np, na, nt = override_alloc.shape
                override_alloc = override_alloc.reshape(bs * np, na, nt)
                repeat_fn = lambda x: x.repeat_interleave(np, dim=0)
            entity2task = repeat_fn(entity2task)
            entity2task[:, :self.args.n_agents] = override_alloc
        x1_ent = self.in_fc_ent(entities)
        x1_alloc = self.in_fc_alloc(entity2task)
        x1_count = self.count_embed(entity2task)
        x1 = F.relu(repeat_fn(x1_ent) + x1_alloc + x1_count)
        x2 = F.relu(self.attn(x1, pre_mask=repeat_fn(attn_mask),
                              post_mask=repeat_fn(entity_mask[:, :self.args.n_agents])))
        out_shape = (bs, self.out_dim)
        if multi_eval:
            out_shape = (bs, np, self.out_dim)
        out = self.out_fc(x2.mean(dim=1)).reshape(*out_shape)
        if self.use_popart:
            if multi_eval:
                out = out * self.popart_weight.unsqueeze(1) + self.popart_bias.unsqueeze(1)
            else:
                out = out * self.popart_weight + self.popart_bias
        if calc_stats:
            return out, {}
        return out


class A2CAllocCritic(nn.Module):
    """
    A2C Critic网络：评估状态值V(s)
    
    特点：
    1. 输入：状态（实体观察）
    2. 输出：状态值V(s)
    """
    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        
        # 特征提取（与Actor共享部分结构）
        self.pi_pointer_net = self.args.hier_agent.get('pi_pointer_net', True)
        
        if not self.pi_pointer_net:
            input_shape += args.n_tasks
        
        self.fc1 = nn.Linear(input_shape, args.alloc_embed_dim)
        self.task_embed = TaskEmbedder(args.alloc_embed_dim, args)
        self.attn = EntityAttentionLayer(
            args.alloc_embed_dim,
            args.alloc_embed_dim,
            args.alloc_embed_dim,
            args,
            n_heads=args.alloc_n_heads,
            use_layernorm=False
        )
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(args.alloc_embed_dim, args.alloc_embed_dim),
            nn.LayerNorm(args.alloc_embed_dim),
            nn.ReLU(),
            nn.Linear(args.alloc_embed_dim, args.alloc_embed_dim)
        )
        
        # V值输出
        self.value_head = nn.Sequential(
            nn.Linear(args.alloc_embed_dim, args.alloc_embed_dim),
            nn.ReLU(),
            nn.Linear(args.alloc_embed_dim, 1)
        )
        
        # PopArt支持（可选）
        self.use_popart = args.popart
        if self.use_popart:
            self.targ_rms = ExponentialMeanStd(alpha=0.01)
            self.popart_weight = nn.Parameter(th.ones(1, 1))
            self.popart_bias = nn.Parameter(th.zeros(1, 1))
    
    def load_state_dict(self, state_dict):
        if self.use_popart:
            targ_rms_state_dict, state_dict = state_dict
            self.targ_rms.load_state_dict(targ_rms_state_dict)
        super().load_state_dict(state_dict)
    
    def state_dict(self):
        if self.use_popart:
            return self.targ_rms.state_dict(), super().state_dict()
        return super().state_dict()
    
    def forward(self, batch, actions=None, calc_stats=False):
        """
        评估状态值V(s)
        
        Args:
            batch: 包含状态信息的batch
            actions: 可选，A2C主要使用V(s)，但保留接口兼容性
        
        Returns:
            values: (bs, 1) - 状态值
        """
        entities = batch['entities']
        entity_mask = batch['entity_mask']
        
        # 特征提取
        entity2task = 1 - batch['entity2task_mask'].float()
        entities_input = entities
        if not self.pi_pointer_net:
            entities_input = th.cat([entities, entity2task], dim=-1)
        
        x1 = self.fc1(entities_input)
        if self.pi_pointer_net:
            x1 += self.task_embed(entity2task)
        
        # 注意力聚合
        attn_mask = groupmask2attnmask(entity_mask)
        x2 = self.attn(
            F.relu(x1),
            pre_mask=attn_mask,
            post_mask=entity_mask
        )
        
        # 状态编码（只关注智能体）
        agent_features = x2[:, :self.args.n_agents]  # (bs, na, embed_dim)
        state_features = self.state_encoder(agent_features)  # (bs, na, embed_dim)
        
        # 聚合智能体特征（平均池化）
        state_embed = state_features.mean(dim=1)  # (bs, embed_dim)
        
        # 输出V值
        values = self.value_head(state_embed)  # (bs, 1)
        
        # 检查NaN并处理（防止训练崩溃）
        if th.isnan(values).any():
            print(f"Warning: NaN detected in Critic values. Replacing with zeros.")
            values = th.where(th.isnan(values), th.zeros_like(values), values)
        
        if self.use_popart:
            values = values * self.popart_weight + self.popart_bias
            # 再次检查NaN（在PopArt之后）
            if th.isnan(values).any():
                print(f"Warning: NaN detected in Critic values after PopArt. Replacing with zeros.")
                values = th.where(th.isnan(values), th.zeros_like(values), values)
        
        if calc_stats:
            return values, {}
        return values
    
    def popart_update(self, targets, mask):
        """PopArt更新（如果启用）"""
        if not self.use_popart:
            return targets
        
        if self.targ_rms.mean is not None:
            old_mean = self.targ_rms.mean.clone()
            old_var = self.targ_rms.var.clone()
            self.targ_rms.update(targets, mask)
        else:
            self.targ_rms.update(targets, mask)
            old_mean = self.targ_rms.mean.clone()
            old_var = self.targ_rms.var.clone()
        
        sd_ratio = old_var.sqrt() / self.targ_rms.var.sqrt()
        self.popart_weight.data.mul_(sd_ratio)
        self.popart_bias.data.mul_(sd_ratio).add_(
            (old_mean - self.targ_rms.mean) / self.targ_rms.var.sqrt()
        )
        return (targets - self.targ_rms.mean) / self.targ_rms.var.sqrt()
    
    def denormalize(self, v):
        """反归一化V值"""
        if self.targ_rms.mean is None or not self.use_popart:
            return v
        return v * self.targ_rms.var.sqrt() + self.targ_rms.mean
