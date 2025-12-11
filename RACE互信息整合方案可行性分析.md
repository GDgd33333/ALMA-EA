# 将RACE互信息机制整合到ALMA-EA的方案

## 目录
1. [整合可行性分析](#整合可行性分析)
2. [整合方案设计](#整合方案设计)
3. [具体实现步骤](#具体实现步骤)
4. [代码实现示例](#代码实现示例)
5. [预期效果与注意事项](#预期效果与注意事项)

---

## 整合可行性分析

### ✅ 可行性：**高度可行**

#### 相似点
1. **都有隐藏状态表示**: 
   - RACE: 智能体的RNN隐藏状态
   - ALMA-EA: EntityBase输出的隐藏状态 `(bs, ts, n_agents, rnn_hidden_dim)`

2. **都有全局信息**:
   - RACE: 全局状态 `state`
   - ALMA-EA: 所有实体信息 `entities`，可以聚合为全局状态

3. **都面临部分可观察问题**:
   - RACE: 通过互信息最大化增强全局理解
   - ALMA-EA: 通过obs_mask限制可见性，互信息可以进一步帮助

#### 整合优势
1. **增强全局理解**: 在部分可观察环境中，帮助智能体更好地理解全局状态
2. **改善任务分配**: 任务嵌入与全局信息的互信息可以改善任务分配质量
3. **提升协作**: 智能体之间通过共享的全局信息理解实现更好的协作

---

## 整合方案设计

### 方案1: 智能体隐藏状态 ↔ 全局实体状态互信息

**目标**: 最大化智能体隐藏状态与全局实体状态之间的互信息

**适用场景**: 
- 提升智能体对全局环境的理解
- 在部分可观察环境中特别有效

**实现位置**:
- 在 `EntityBase` 的输出（隐藏状态）和全局实体聚合之间

### 方案2: 任务嵌入 ↔ 全局任务状态互信息

**目标**: 最大化任务嵌入与全局任务状态之间的互信息

**适用场景**:
- 改善任务分配质量
- 使任务嵌入包含更多全局上下文信息

**实现位置**:
- 在 `AllocationPolicy` 的任务嵌入生成过程中

### 方案3: 智能体嵌入 ↔ 全局任务信息互信息（推荐）

**目标**: 最大化智能体嵌入与全局任务信息之间的互信息

**适用场景**:
- 同时提升智能体理解和任务分配质量
- 最符合ALMA-EA的层次化架构

**实现位置**:
- 在提案网络（AllocationPolicy）的智能体嵌入和任务嵌入之间

---

## 具体实现步骤

### 步骤1: 添加MINE网络模块

**文件**: `src/modules/agents/mutual_info.py` (新建)

```python
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math

def get_negative_expectation(q_samples, measure, average=True):
    """计算负样本期望"""
    log_2 = math.log(2.)
    if measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'KL':
        q_samples = th.clamp(q_samples, -1e6, 9.5)
        Eq = th.exp(q_samples - 1.)
    else:
        raise ValueError(f"Unknown measure: {measure}")
    return Eq.mean() if average else Eq

def get_positive_expectation(p_samples, measure, average=True):
    """计算正样本期望"""
    log_2 = math.log(2.)
    if measure == 'JSD':
        Ep = log_2 - F.softplus(-p_samples)
    elif measure == 'GAN':
        Ep = -F.softplus(-p_samples)
    elif measure == 'KL':
        Ep = p_samples
    else:
        raise ValueError(f"Unknown measure: {measure}")
    return Ep.mean() if average else Ep

def fenchel_dual_loss(l, m, measure="JSD"):
    """
    计算Fenchel对偶损失来估计互信息
    
    Args:
        l: 局部特征 (智能体嵌入) - shape: (N, embed_dim)
        m: 全局特征 (全局状态) - shape: (N, embed_dim)
        measure: f-散度度量方式
    """
    N, units = l.size()
    # 计算相似度矩阵
    u = th.mm(m, l.t())  # (N, N)
    
    # 创建掩码矩阵
    mask = th.eye(N, device=l.device)  # 正样本掩码
    n_mask = 1 - mask  # 负样本掩码
    
    # 计算正负样本期望
    E_pos = get_positive_expectation(u, measure, average=False)
    E_neg = get_negative_expectation(u, measure, average=False)
    
    # 提取正负样本项
    E_pos_term = (E_pos * mask).sum(1)
    E_neg_term = (E_neg * n_mask).sum(1) / (N - 1)
    
    # 互信息损失
    loss = E_neg_term - E_pos_term
    
    # 互信息估计值
    MI = (E_pos * mask).sum(1)
    
    return loss, MI

class MINE(nn.Module):
    """互信息神经估计网络"""
    def __init__(self, x_dim, y_dim, measure="JSD", embed_dim=128):
        super(MINE, self).__init__()
        self.measure = measure
        self.x_dim = x_dim
        self.y_dim = y_dim
        
        # 编码局部特征（智能体嵌入）
        self.l1 = nn.Linear(x_dim, embed_dim)
        self.l2 = nn.Linear(embed_dim, embed_dim)
        
        # 编码全局特征（全局状态）
        self.l3 = nn.Linear(y_dim, embed_dim)
        
        self.nonlinearity = F.leaky_relu
    
    def forward(self, x, y):
        """
        Args:
            x: 局部特征 (智能体嵌入) - shape: (N, x_dim)
            y: 全局特征 (全局状态) - shape: (N, y_dim)
        Returns:
            loss: 互信息损失
            MI: 互信息估计值
        """
        # 编码局部特征
        em_1 = self.nonlinearity(self.l1(x), inplace=True)
        em_1 = self.nonlinearity(self.l2(em_1), inplace=True)
        
        # 编码全局特征
        em_2 = self.nonlinearity(self.l3(y), inplace=True)
        
        # 计算互信息
        loss, MI = fenchel_dual_loss(em_1, em_2, measure=self.measure)
        return loss, MI
```

### 步骤2: 在AllocationPolicy中集成互信息

**文件**: `src/modules/agents/allocation_policies.py`

**修改点1: 初始化MINE网络**

```python
# 在 __init__ 方法中添加
from .mutual_info import MINE

class AutoregressiveAllocPolicy(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        # ... 原有代码 ...
        
        # 添加互信息网络
        if getattr(args.hier_agent, 'use_mutual_info', False):
            # 智能体嵌入维度
            agent_embed_dim = args.attn_embed_dim
            # 全局状态维度：所有实体的聚合
            global_state_dim = args.attn_embed_dim  # 或使用其他维度
            self.MINE = MINE(
                x_dim=agent_embed_dim,
                y_dim=global_state_dim,
                measure=getattr(args.hier_agent, 'mi_measure', 'JSD'),
                embed_dim=getattr(args.hier_agent, 'mi_embed_dim', 128)
            )
            self.use_mutual_info = True
        else:
            self.use_mutual_info = False
```

**修改点2: 在forward中计算互信息**

```python
def forward(self, batch, calc_stats=False, test_mode=False, n_proposals=-1):
    # ... 原有代码到第220行 ...
    
    # get agent embeddings
    agent_x1 = x1[:, :nag]
    if self.pi_ag_attn:
        # ... 原有注意力代码 ...
        agent_embeds = agent_x1 + self.attn(...)
    else:
        agent_embeds = agent_x1
    
    # ========== 新增：计算互信息 ==========
    mi_loss = 0
    mi_value = 0
    if self.use_mutual_info and not test_mode:
        # 构建全局状态：聚合所有实体信息
        # 方法1: 使用所有实体的平均
        global_state = x1.mean(dim=1)  # (bs, attn_embed_dim)
        
        # 方法2: 使用任务嵌入的聚合（更符合任务分配场景）
        # global_state = task_x2.mean(dim=1)  # (bs, attn_embed_dim)
        
        # 对每个智能体计算互信息
        bs = agent_embeds.shape[0]
        for ai in range(nag):
            agent_embed = agent_embeds[:, ai, :]  # (bs, attn_embed_dim)
            
            # 重复全局状态以匹配智能体数量
            repeat_global = global_state  # (bs, attn_embed_dim)
            
            # 计算互信息
            loss, MI = self.MINE(agent_embed, repeat_global)
            mi_loss += loss.mean()
            mi_value += MI.mean()
        
        mi_loss = mi_loss / nag
        mi_value = mi_value / nag
    
    # ... 继续原有代码 ...
    
    if calc_stats and self.use_mutual_info:
        stats['mi_loss'] = mi_loss.item() if isinstance(mi_loss, th.Tensor) else mi_loss
        stats['mi_value'] = mi_value.item() if isinstance(mi_value, th.Tensor) else mi_value
    
    # ... 返回 ...
```

### 步骤3: 在学习器中添加互信息损失

**文件**: `src/learners/q_learner.py`

**修改点: 在train方法中添加互信息损失**

```python
def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
    # ... 原有代码 ...
    
    # ========== 新增：计算互信息损失 ==========
    mi_loss = 0
    if (self.args.hier_agent["task_allocation"] == "aql" and 
        getattr(self.args.hier_agent, 'use_mutual_info', False)):
        
        # 获取提案网络的互信息损失
        meta_batch = self._make_meta_batch(batch)
        allocs, pi_stats = self.mac.compute_allocation(
            meta_batch, 
            calc_stats=True, 
            test_mode=False
        )
        
        if 'mi_loss' in pi_stats:
            mi_loss = pi_stats['mi_loss']
    
    # ========== 修改：将互信息损失加入总损失 ==========
    # 原有的策略损失
    pi_loss = (amort_loss + 
               self.args.hier_agent['entropy_loss'] * entropy_loss)
    
    # 添加互信息损失
    if getattr(self.args.hier_agent, 'use_mutual_info', False):
        mi_weight = getattr(self.args.hier_agent, 'mi_weight', 0.001)
        pi_loss = pi_loss + mi_weight * mi_loss
    
    # ... 继续原有代码 ...
```

### 步骤4: 添加配置参数

**文件**: `src/config/algs/qmix_atten_ea.yaml` 或 `default.yaml`

```yaml
hier_agent:
  # ... 原有配置 ...
  
  # 互信息相关配置
  use_mutual_info: True          # 是否使用互信息
  mi_measure: "JSD"             # 互信息度量方式: JSD, GAN, KL
  mi_embed_dim: 128             # MINE网络的嵌入维度
  mi_weight: 0.001              # 互信息损失权重（对应RACE的state_alpha）
```

---

## 代码实现示例

### 完整示例：在AllocationPolicy中集成

```python
# src/modules/agents/allocation_policies.py

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .mutual_info import MINE  # 新增导入

class AutoregressiveAllocPolicy(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        # ... 原有初始化代码 ...
        
        # ========== 新增：互信息网络 ==========
        if getattr(args.hier_agent, 'use_mutual_info', False):
            agent_embed_dim = args.attn_embed_dim
            # 全局状态维度：使用任务嵌入维度
            global_state_dim = args.attn_embed_dim
            self.MINE = MINE(
                x_dim=agent_embed_dim,
                y_dim=global_state_dim,
                measure=getattr(args.hier_agent, 'mi_measure', 'JSD'),
                embed_dim=getattr(args.hier_agent, 'mi_embed_dim', 128)
            )
            self.use_mutual_info = True
        else:
            self.use_mutual_info = False
    
    def forward(self, batch, calc_stats=False, test_mode=False, n_proposals=-1):
        # ... 原有代码到获取agent_embeds和task_x2 ...
        
        # ========== 新增：计算互信息 ==========
        mi_loss = 0
        mi_value = 0
        if self.use_mutual_info and not test_mode:
            bs = agent_embeds.shape[0]
            nag = self.args.n_agents
            
            # 构建全局状态：使用任务嵌入的聚合
            # task_x2 shape: (bs, n_tasks, attn_embed_dim)
            global_state = task_x2.mean(dim=1)  # (bs, attn_embed_dim)
            
            # 对每个智能体计算互信息
            for ai in range(nag):
                agent_embed = agent_embeds[:, ai, :]  # (bs, attn_embed_dim)
                
                # 计算互信息
                loss, MI = self.MINE(agent_embed, global_state)
                mi_loss += loss.mean()
                mi_value += MI.mean()
            
            mi_loss = mi_loss / nag
            mi_value = mi_value / nag
        
        # ... 继续原有代码生成allocs ...
        
        if calc_stats:
            stats = {}
            # ... 原有统计 ...
            if self.use_mutual_info:
                stats['mi_loss'] = mi_loss.item() if isinstance(mi_loss, th.Tensor) else mi_loss
                stats['mi_value'] = mi_value.item() if isinstance(mi_value, th.Tensor) else mi_value
            return allocs, stats
        
        return allocs
```

### 在学习器中集成

```python
# src/learners/q_learner.py

def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
    # ... 原有代码计算q_loss ...
    
    # ========== 任务分配策略训练 ==========
    if self.args.hier_agent["task_allocation"] == "aql":
        # ... 原有代码计算pi_loss ...
        
        # ========== 新增：添加互信息损失 ==========
        if getattr(self.args.hier_agent, 'use_mutual_info', False):
            # 获取互信息损失（已在compute_allocation中计算）
            meta_batch = self._make_meta_batch(batch)
            allocs, pi_stats = self.mac.compute_allocation(
                meta_batch, calc_stats=True, test_mode=False
            )
            
            if 'mi_loss' in pi_stats:
                mi_weight = getattr(self.args.hier_agent, 'mi_weight', 0.001)
                mi_loss = pi_stats['mi_loss']
                pi_loss = pi_loss + mi_weight * mi_loss
                
                # 记录统计信息
                if t_env - self.log_alloc_stats_t >= self.args.learner_log_interval:
                    self.logger.log_stat("losses/mi_loss", mi_loss, t_env)
                    self.logger.log_stat("mi_value", pi_stats.get('mi_value', 0), t_env)
        
        # ... 继续原有代码 ...
```

---

## 预期效果与注意事项

### 预期效果

1. **提升全局理解**: 
   - 智能体更好地理解全局环境状态
   - 在部分可观察环境中表现更优

2. **改善任务分配**:
   - 任务嵌入包含更多全局上下文
   - 任务分配更加合理和协调

3. **增强协作**:
   - 智能体之间通过共享的全局信息理解实现更好的协作

### 注意事项

1. **计算开销**:
   - MINE网络增加了一定的计算量
   - 可以通过调整`mi_embed_dim`来控制

2. **超参数调优**:
   - `mi_weight`: 互信息损失权重，建议从0.001开始（对应RACE的state_alpha）
   - `mi_measure`: 建议使用"JSD"（Jensen-Shannon Divergence）

3. **训练稳定性**:
   - 互信息损失可能不稳定，建议使用梯度裁剪
   - 可以只在训练时计算，测试时关闭

4. **全局状态构建**:
   - 需要选择合适的全局状态表示方式
   - 选项1: 所有实体的平均
   - 选项2: 任务嵌入的聚合（推荐，更符合任务分配场景）
   - 选项3: 使用专门的全局状态编码器

5. **与现有机制的协调**:
   - 注意与COPA、注意力机制等的协调
   - 避免信息冗余

### 实验建议

1. **消融实验**:
   - 对比有无互信息的性能差异
   - 测试不同的全局状态构建方式

2. **超参数搜索**:
   - `mi_weight`: [0.0001, 0.001, 0.01]
   - `mi_embed_dim`: [64, 128, 256]

3. **环境测试**:
   - 在部分可观察性强的环境中测试效果更明显
   - 对比不同观察范围下的性能提升

---

## 总结

将RACE的互信息机制整合到ALMA-EA是**高度可行**的，主要优势：

1. ✅ **架构兼容**: 两者都有隐藏状态和全局信息
2. ✅ **目标一致**: 都旨在提升部分可观察环境下的性能
3. ✅ **实现简单**: MINE网络结构简单，易于集成
4. ✅ **效果预期**: 可以同时提升智能体理解和任务分配质量

**推荐实现顺序**:
1. 先实现方案3（智能体嵌入 ↔ 全局任务信息）
2. 验证效果后再考虑方案1和方案2
3. 根据实验结果调整超参数和全局状态构建方式

