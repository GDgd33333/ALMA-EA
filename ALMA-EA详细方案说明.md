# ALMA-EA 详细方案说明文档

## 目录
1. [方案概述](#方案概述)
2. [系统架构](#系统架构)
3. [核心组件详解](#核心组件详解)
4. [进化算法实现](#进化算法实现)
5. [双向同步机制](#双向同步机制)
6. [训练流程](#训练流程)
7. [配置参数详解](#配置参数详解)
8. [关键技术细节](#关键技术细节)
9. [性能优化](#性能优化)
10. [使用指南](#使用指南)

---

## 方案概述

### 1.1 项目背景

ALMA-EA 是基于 ALMA (Hierarchical Learning for Composite Multi-Agent Tasks) 框架的增强版本，通过引入进化算法（Evolutionary Algorithm, EA）来优化任务分配策略（Allocation Policy）。该方案将强化学习（RL）与进化算法相结合，实现了一种混合优化方法。

### 1.2 核心思想

- **分层学习**：采用两层架构，高层负责任务分配（Task Allocation），低层负责具体动作执行
- **混合优化**：RL 负责策略梯度优化，EA 负责全局搜索和探索
- **双向同步**：RL 网络和 EA 种群之间相互学习，实现知识交换

### 1.3 主要特点

1. **串行评估**：EA 种群评估采用串行方式，避免嵌套多进程问题
2. **阈值控制双向同步**：RL 网络和 EA 种群之间的同步都通过阈值控制，只有当一方明显优于另一方时才进行同步
   - **RL→EA 同步**：只有当主网络明显优于 EA 最差个体时（超过阈值），才替换 EA 最差个体
   - **EA→RL 同步**：只有当 EA 最优个体明显优于主网络时（超过阈值），才同步到主网络
3. **精英保留**：每代保留最优个体，避免性能退化

---

## 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                     训练主循环 (run.py)                      │
│  ┌──────────────┐      ┌──────────────┐                   │
│  │  Runner      │      │  Learner      │                   │
│  │  (环境交互)   │◄────►│  (RL训练)     │                   │
│  └──────────────┘      └──────────────┘                   │
│         │                      │                            │
│         │                      │                            │
│         ▼                      ▼                            │
│  ┌──────────────────────────────────────┐                 │
│  │      EntityMAC (多智能体控制器)        │                 │
│  │  ┌──────────────┐  ┌──────────────┐ │                 │
│  │  │ alloc_policy │  │ alloc_critic │ │                 │
│  │  │ (提案网络)    │  │ (打分网络)   │ │                 │
│  │  └──────────────┘  └──────────────┘ │                 │
│  └──────────────────────────────────────┘                 │
│         │                                                  │
│         │ 每 sync_interval 步                              │
│         ▼                                                  │
│  ┌──────────────────────────────────────┐                 │
│  │      EAManager (进化算法管理器)        │                 │
│  │  ┌──────────────────────────────────┐ │                 │
│  │  │  Population (种群)                │ │                 │
│  │  │  ┌────┐ ┌────┐ ┌────┐ ┌────┐    │ │                 │
│  │  │  │G1  │ │G2  │ │G3  │ │G4  │... │ │                 │
│  │  │  └────┘ └────┘ └────┘ └────┘    │ │                 │
│  │  └──────────────────────────────────┘ │                 │
│  │  ┌──────────────────────────────────┐ │                 │
│  │  │  进化算子                         │ │                 │
│  │  │  - Selection (选择)               │ │                 │
│  │  │  - Crossover (交叉)               │ │                 │
│  │  │  - Mutation (变异)                │ │                 │
│  │  └──────────────────────────────────┘ │                 │
│  └──────────────────────────────────────┘                 │
│         │                      ▲                            │
│         │  双向同步             │                            │
│         └──────────────────────┘                            │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 模块划分

#### 2.2.1 核心模块

1. **`src/run.py`**: 训练主循环，协调所有组件
2. **`src/ea/ea_manager.py`**: 进化算法管理器，负责种群管理和进化操作
3. **`src/ea/genome.py`**: 基因组类，封装分配策略网络参数
4. **`src/ea/operators.py`**: 进化算子（选择、交叉、变异）
5. **`src/controllers/entity_controller.py`**: 多智能体控制器
6. **`src/modules/agents/allocation_policies.py`**: 分配策略网络（AutoregressiveAllocPolicy）

#### 2.2.2 支持模块

- **Runners**: `episode_runner.py`, `parallel_runner.py` - 环境运行器
- **Learners**: `q_learner.py` - Q学习器
- **Mixers**: `qmix.py`, `flex_qmix.py` - 值函数混合网络
- **Environments**: `starcraft2.py`, `firefighters.py` - 环境实现

---

## 核心组件详解

### 3.1 AllocationGenome (基因组)

#### 3.1.1 类定义

```python
class AllocationGenome:
    """分配器基因编码类，封装分配器网络参数作为进化个体"""
```

#### 3.1.2 核心属性

- **`genome_id`**: 基因组唯一标识符
- **`alloc_policy`**: 分配策略网络（AutoregressiveAllocPolicy）的深拷贝
- **`parameters`**: 参数字典，存储网络参数的副本
- **`fitness`**: 当前适应度值（累加值）
- **`fitness_history`**: 适应度历史记录列表
- **`evaluation_count`**: 评估次数计数器

#### 3.1.3 关键方法

1. **`_extract_parameters()`**: 提取网络参数到字典中
   - 遍历 `alloc_policy` 的所有命名参数
   - 将参数数据克隆到 `parameters` 字典

2. **`update_parameters(new_params)`**: 更新网络参数
   - 更新参数字典中的参数
   - 同步到实际网络

3. **`get_parameters()`**: 获取参数字典的深拷贝

4. **`clone()`**: 克隆基因组，创建新实例并复制所有属性

5. **`reset_fitness()`**: 重置适应度值和评估次数

6. **`add_fitness(reward)`**: 累加奖励到总适应度

7. **`get_average_fitness()`**: 计算平均适应度（总适应度/评估次数）

8. **`update_fitness_history()`**: 更新适应度历史记录

#### 3.1.4 参数向量化方法

- **`get_parameter_vector()`**: 将所有参数展平为一维向量
- **`set_parameter_vector(param_vector)`**: 从一维向量恢复参数

### 3.2 EAManager (进化算法管理器)

#### 3.2.1 初始化参数

```python
def __init__(self,
    population_size: int = 5,          # 种群大小
    elite_size: int = 2,               # 精英个体数量
    mutation_rate: float = 0.1,        # 变异率（10%）
    mutation_strength: float = 0.1,    # 变异强度
    crossover_rate: float = 0.8,        # 交叉率（80%）
    selection_type: str = "tournament", # 选择类型
    tournament_size: int = 3,           # 锦标赛大小
    evaluation_episodes: int = 5,       # 评估回合数
    sync_interval: int = 1000,         # 同步间隔
    device: str = "cuda",              # 计算设备
    sync_threshold: float = 0.05,      # 同步阈值（5%）
    enable_bidirectional_sync: bool = True  # 启用双向同步
)
```

#### 3.2.2 核心属性

- **`population`**: 种群列表，存储所有 `AllocationGenome` 个体
- **`generation`**: 当前代数
- **`best_genome`**: 历史最优个体
- **`fitness_history`**: 种群适应度历史
- **`main_network_fitness_history`**: 主网络适应度历史
- **`sync_stats`**: 同步统计信息
  - `'ea_to_main_syncs'`: EA→RL 同步次数
  - `'main_to_ea_syncs'`: RL→EA 同步次数
  - `'no_ea_to_main_sync'`: EA 未同步到 RL 的次数

#### 3.2.3 核心方法

1. **`initialize_population(base_alloc_policy)`**
   - 清空现有种群
   - 创建 `population_size` 个个体
   - 每个个体基于 `base_alloc_policy` 深拷贝创建
   - 将第一个个体作为初始最优个体

2. **`evaluate_genome(genome, runner, t_env)`**
   - 临时替换主网络的 `alloc_policy` 为当前基因组的 `alloc_policy`
   - 运行 `evaluation_episodes` 个评估回合
   - 累加奖励到基因组适应度
   - 恢复原始 `alloc_policy`
   - 返回平均适应度

3. **`evaluate_population(runner, t_env)`**
   - **串行评估**：遍历种群中的每个个体
   - 重置个体适应度
   - 调用 `evaluate_genome` 评估个体
   - 更新个体适应度历史
   - 更新种群统计信息
   - 更新最优个体

4. **`evaluate_main_network_performance(runner, t_env)`**
   - 评估主网络（RL）的性能
   - 运行 `evaluation_episodes` 个评估回合
   - 计算平均适应度
   - 记录到 `main_network_fitness_history`

5. **`evolve_generation(main_network_fitness, main_alloc_policy)`**
   - 如果启用双向同步，调用 `_bidirectional_sync`
   - 选择父代（种群大小 - 精英数量）
   - 精英保留：选择适应度最高的 `elite_size` 个个体
   - 交叉和变异生成新个体
   - 更新种群和代数

### 3.3 AutoregressiveAllocPolicy (自回归分配策略网络)

#### 3.3.1 网络结构

```
输入: entities, task_mask, entity_mask, avail_actions
  │
  ▼
FC1 (input_shape → attn_embed_dim)
  │
  ▼
TaskEmbedder (任务嵌入)
  │
  ▼
EntityAttentionLayer (实体注意力层)
  │
  ▼
[自回归循环]
  │
  ├─→ Agent 0: 选择任务分配
  ├─→ Agent 1: 选择任务分配（考虑 Agent 0 的选择）
  ├─→ Agent 2: 选择任务分配（考虑 Agent 0,1 的选择）
  └─→ ...
  │
  ▼
输出: allocations (batch_size, n_agents, n_tasks)
```

#### 3.3.2 关键特性

1. **自回归生成**：逐个智能体生成任务分配，每个智能体的选择依赖于前面智能体的选择
2. **Pointer Network**：使用点积注意力机制选择任务
3. **任务掩码**：只考虑活跃任务（task_mask）
4. **实体掩码**：只考虑活跃智能体（entity_mask）

#### 3.3.3 核心方法

- **`_autoreg_forward()`**: 自回归前向传播
  - 循环处理每个智能体
  - 计算当前智能体对每个任务的 logits
  - 应用任务掩码和智能体掩码
  - 采样任务分配
  - 更新任务计数

### 3.4 EntityMAC (多智能体控制器)

#### 3.4.1 核心组件

- **`alloc_policy`**: AutoregressiveAllocPolicy 实例（提案网络）
- **`alloc_critic`**: AllocationCritic 实例（打分网络）

#### 3.4.2 关键方法

1. **`compute_allocation(meta_batch, ...)`**
   - 调用 `alloc_policy` 生成 `n_proposals` 个提案
   - 调用 `alloc_critic` 评估每个提案
   - 选择得分最高的提案
   - 在训练模式下应用 epsilon-greedy 探索

2. **`evaluate_allocation(meta_batch, ...)`**
   - 调用 `alloc_critic` 评估给定的任务分配

---

## 进化算法实现

### 4.1 选择算子 (SelectionOperator)

#### 4.1.1 锦标赛选择 (Tournament Selection)

```python
def _tournament_selection(self, population, num_parents):
    parents = []
    for _ in range(num_parents):
        # 随机选择 tournament_size 个个体
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        # 选择适应度最高的
        winner = max(tournament, key=lambda x: x.get_average_fitness())
        parents.append(winner)
    return parents
```

**特点**：
- 随机性：每次随机选择 `tournament_size` 个个体
- 选择压力：适应度高的个体更容易被选中
- 参数：`tournament_size = 3`（默认）

#### 4.1.2 精英选择 (Elite Selection)

```python
def _elite_selection(self, population, num_parents):
    sorted_pop = sorted(population, key=lambda x: x.get_average_fitness(), reverse=True)
    return sorted_pop[:num_parents]
```

**特点**：
- 确定性：总是选择适应度最高的个体
- 无随机性：结果可重复

#### 4.1.3 轮盘赌选择 (Roulette Wheel Selection)

```python
def _roulette_selection(self, population, num_parents):
    fitnesses = [max(0.001, x.get_average_fitness()) for x in population]
    total_fitness = sum(fitnesses)
    probabilities = [f / total_fitness for f in fitnesses]
    # 根据概率随机选择
```

**特点**：
- 概率与适应度成正比
- 适应度高的个体被选中的概率大

### 4.2 变异算子 (MutationOperator)

#### 4.2.1 标准变异

```python
def mutate(self, genome):
    mutated_genome = genome.clone()
    mutated_params = {}
    
    for name, param in genome.parameters.items():
        if random.random() < self.mutation_rate:  # 10% 概率
            # 高斯噪声变异
            noise = th.randn_like(param) * self.mutation_strength  # 标准差 = 0.1
            mutated_params[name] = param + noise
        else:
            mutated_params[name] = param.clone()
    
    mutated_genome.update_parameters(mutated_params)
    return mutated_genome
```

**参数**：
- `mutation_rate = 0.1`: 10% 的参数发生变异
- `mutation_strength = 0.1`: 变异噪声的标准差

**特点**：
- 高斯噪声：`N(0, mutation_strength^2)`
- 独立变异：每个参数独立决定是否变异

#### 4.2.2 自适应变异

```python
def adaptive_mutate(self, genome, generation, max_generations):
    progress = generation / max_generations
    adaptive_strength = self.mutation_strength * (1 - progress * 0.5)  # 逐渐减小
    adaptive_rate = self.mutation_rate * (1 - progress * 0.3)  # 逐渐减小
    # ... 使用自适应参数进行变异
```

**特点**：
- 随代数递减变异强度和变异率
- 早期探索，后期利用

### 4.3 交叉算子 (CrossoverOperator)

#### 4.3.1 均匀交叉 (Uniform Crossover)

```python
def _uniform_crossover(self, parent1, parent2):
    child1 = parent1.clone()
    child2 = parent2.clone()
    
    for name in parent1.parameters.keys():
        param1 = parent1.parameters[name]
        param2 = parent2.parameters[name]
        
        # 随机选择每个参数来自哪个父代
        mask = th.rand_like(param1) < 0.5
        child1_params[name] = th.where(mask, param1, param2)
        child2_params[name] = th.where(mask, param2, param1)
    
    return child1, child2
```

**特点**：
- 每个参数独立选择来自哪个父代
- 50% 概率来自 parent1，50% 概率来自 parent2

#### 4.3.2 算术交叉 (Arithmetic Crossover)

```python
def _arithmetic_crossover(self, parent1, parent2):
    alpha = random.uniform(0, 1)  # 交叉系数
    
    for name in parent1.parameters.keys():
        param1 = parent1.parameters[name]
        param2 = parent2.parameters[name]
        
        # 算术交叉：child = alpha * parent1 + (1-alpha) * parent2
        child1_params[name] = alpha * param1 + (1 - alpha) * param2
        child2_params[name] = (1 - alpha) * param1 + alpha * param2
```

**特点**：
- 参数值的线性组合
- 产生介于两个父代之间的子代

### 4.4 进化流程

```
1. 初始化种群
   └─→ 基于主网络的 alloc_policy 创建 population_size 个个体

2. 评估种群（每 sync_interval 步）
   ├─→ 串行评估每个个体
   │   ├─→ 替换主网络的 alloc_policy
   │   ├─→ 运行 evaluation_episodes 个评估回合
   │   ├─→ 累加奖励
   │   └─→ 恢复主网络的 alloc_policy
   └─→ 更新种群统计信息

3. 评估主网络性能
   └─→ 运行 evaluation_episodes 个评估回合

4. 双向同步（如果启用）
   └─→ EA → RL: 如果 EA 最优 > RL + threshold，同步到 RL

5. 进化一代
   ├─→ 选择父代（种群大小 - 精英数量）
   ├─→ 精英保留（保留 elite_size 个最优个体）
   ├─→ 交叉生成新个体
   ├─→ 变异新个体
   └─→ 更新种群和代数

6. 重复步骤 2-5
```

---

## 双向同步机制

### 5.1 同步策略

#### 5.1.1 阈值控制 RL→EA 同步

**策略**：只有当主网络明显优于 EA 最差个体时（超过阈值），才替换 EA 最差个体

**实现**：
```python
if main_network_fitness > worst_ea_fitness + self.sync_threshold:
    # 用主网络替换 EA 最差个体
    self._replace_worst_with_main_network(main_network_fitness, main_alloc_policy)
    self.sync_stats['main_to_ea_syncs'] += 1
else:
    self.sync_stats['no_main_to_ea_sync'] += 1
```

**阈值计算**：
- `sync_threshold = 0.05`（默认）
- **绝对差值**：不是相对百分比
- 例如：如果 `main_network_fitness = 0.60`，`worst_ea_fitness = 0.54`，则 `0.60 > 0.54 + 0.05 = 0.59`，触发同步

**原因**：
- 避免频繁替换，保持 EA 种群的稳定性
- 只有当主网络明显优于 EA 最差个体时才注入新知识
- 减少噪声影响

#### 5.1.2 阈值控制 EA→RL 同步

**策略**：只有当 EA 最优个体明显优于主网络时（超过阈值），才同步到主网络

**实现**：
```python
if best_ea_fitness > main_network_fitness + self.sync_threshold:
    # 同步 EA 最优个体到主网络
    self.sync_best_to_main(main_alloc_policy)
    self.sync_stats['ea_to_main_syncs'] += 1
else:
    self.sync_stats['no_ea_to_main_sync'] += 1
```

**阈值计算**：
- `sync_threshold = 0.05`（默认）
- **绝对差值**：不是相对百分比
- 例如：如果 `best_ea_fitness = 0.60`，`main_network_fitness = 0.54`，则 `0.60 > 0.54 + 0.05 = 0.59`，触发同步

**原因**：
- 避免频繁切换策略，保持稳定性
- 只有当 EA 找到明显更好的策略时才更新 RL
- 减少噪声影响

### 5.2 同步流程

#### 5.2.1 启用双向同步时（`enable_bidirectional_sync=True`）

```
1. 评估 EA 种群性能
   ├─→ best_ea_fitness: EA 最优适应度
   ├─→ avg_ea_fitness: EA 平均适应度
   └─→ worst_ea_fitness: EA 最差适应度

2. 评估主网络性能
   └─→ main_network_fitness: 主网络适应度

3. 阈值控制 RL → EA 同步
   ├─→ 如果 main_network_fitness > worst_ea_fitness + threshold
   │   └─→ 用主网络替换 EA 最差个体
   └─→ 否则，不进行同步

4. 阈值控制 EA → RL 同步
   ├─→ 如果 best_ea_fitness > main_network_fitness + threshold
   │   └─→ 同步 EA 最优到主网络
   └─→ 否则，不进行同步
```

#### 5.2.2 未启用双向同步时（`enable_bidirectional_sync=False`）

```
1. 评估 EA 种群性能
2. 进化到下一代
3. 无条件将 EA 最优个体同步到主网络（不进行阈值判断）
```

**注意**：当 `enable_bidirectional_sync=False` 时，EA→RL 同步是无条件的，不进行阈值判断。这是为了确保 EA 的改进能够及时应用到主网络。
```

### 5.3 阈值影响分析

#### 5.3.1 阈值过小（如 0.01）

**优点**：
- 同步频繁，知识交换活跃
- RL 和 EA 能够快速相互学习
- 双向知识交换及时

**缺点**：
- 可能过于敏感，噪声也会触发同步
- 可能导致策略频繁切换，不稳定
- 计算开销增加
- RL→EA 和 EA→RL 都可能过于频繁

#### 5.3.2 阈值适中（如 0.05，默认值）

**优点**：
- 平衡了同步频率和稳定性
- 既能及时同步重要改进，又能避免噪声干扰
- 双向同步都有合理的触发条件

**缺点**：
- 可能需要根据任务特性调整

#### 5.3.3 阈值过大（如 0.15）

**缺点**：
- 同步较少，知识交换不活跃
- RL→EA 同步减少：主网络的知识难以注入 EA 种群
- EA→RL 同步减少：EA 的探索成果难以同步到主网络
- EA 种群可能陷入局部最优
- RL 无法及时从 EA 的探索中受益
- 双向知识交换受阻

### 5.4 双向同步的实际效果

#### 5.4.1 同步统计信息

每次评估后，可以通过 `print_stats()` 查看同步统计：

```
=== Bidirectional Sync Stats ===
EA → Main syncs: 15              # EA 同步到主网络的次数
Main → EA syncs: 12              # 主网络同步到 EA 的次数
No EA→Main sync (threshold not met): 5   # EA 未达到阈值，未同步的次数
No Main→EA sync (threshold not met): 8   # 主网络未达到阈值，未同步的次数
Sync threshold: 0.0500
```

**解读**：
- 如果 `EA → Main syncs` 很少，说明 EA 种群性能提升有限，可能需要调整 EA 参数
- 如果 `Main → EA syncs` 很少，说明主网络改进有限，或者 EA 种群已经很好
- 如果两个方向的同步都很少，可能需要降低 `sync_threshold`

#### 5.4.2 同步频率建议

- **理想情况**：每个方向每 2-3 次评估同步一次（约 30-50% 的同步率）
- **同步过少**：如果同步率 < 20%，考虑降低阈值
- **同步过多**：如果同步率 > 80%，考虑提高阈值

#### 5.4.3 双向同步的优势

1. **知识互补**：
   - RL 通过梯度下降持续优化，其知识对 EA 有价值
   - EA 通过全局搜索可能找到 RL 难以发现的策略
   - 双向同步确保两者相互学习

2. **稳定性**：
   - 阈值控制避免频繁切换，保持策略稳定性
   - 只有当一方明显优于另一方时才同步，减少噪声影响

3. **探索与利用平衡**：
   - EA 负责探索，RL 负责利用
   - 双向同步实现探索与利用的动态平衡

---

## 训练流程

### 6.1 主训练循环

```python
while runner.t_env <= args.t_max:
    # 1. 运行一个 episode
    episode_batch, _ = runner.run(test_mode=False)
    buffer.insert_episode_batch(episode_batch)
    
    # 2. 如果缓冲区有足够样本，进行 RL 训练
    if buffer.can_sample(args.batch_size):
        for _ in range(args.training_iters):
            episode_sample = buffer.sample(args.batch_size)
            learner.train(episode_sample, runner.t_env, episode)
    
    # 3. 如果使用 AQL，进行分配策略训练
    if args.hier_agent["task_allocation"] == "aql":
        # ... 分配策略训练代码 ...
        learner.alloc_train_aql(alloc_episode_sample, runner.t_env, episode)
    
    # 4. 执行进化算法评估和进化（每 sync_interval 步）
    if ea_manager is not None and (runner.t_env - last_ea_eval_T) >= ea_manager.sync_interval:
        # 4.1 评估种群
        ea_manager.evaluate_population(runner, runner.t_env)
        
        # 4.2 评估主网络性能
        if ea_manager.enable_bidirectional_sync:
            main_network_fitness = ea_manager.evaluate_main_network_performance(runner, runner.t_env)
        
        # 4.3 进化到下一代（包含双向同步）
        ea_manager.evolve_generation(main_network_fitness=main_network_fitness, 
                                     main_alloc_policy=mac.alloc_policy)
        
        # 4.4 如果未启用双向同步，同步最优个体到主网络
        if not ea_manager.enable_bidirectional_sync:
            ea_manager.sync_best_to_main(mac.alloc_policy)
        
        last_ea_eval_T = runner.t_env
    
    # 5. 定期测试
    if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
        for _ in range(n_test_runs):
            runner.run(test_mode=True)
    
    # 6. 定期保存模型
    if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval):
        learner.save_models(save_path_base)
```

### 6.2 时间线示例

假设 `sync_interval = 10000`，`t_max = 2000000`：

```
t_env = 0
  └─→ 初始化 EA 种群

t_env = 10000
  ├─→ 评估 EA 种群（串行，5 个个体 × 5 个 episode = 25 个 episode）
  ├─→ 评估主网络（5 个 episode）
  ├─→ 双向同步
  │   ├─→ RL → EA: 如果主网络 > EA最差 + 0.05，替换最差个体
  │   └─→ EA → RL: 如果 EA 最优 > 主网络 + 0.05，同步到主网络
  └─→ 进化一代

t_env = 20000
  ├─→ 评估 EA 种群
  ├─→ 评估主网络
  ├─→ 双向同步
  └─→ 进化一代

...

t_env = 2000000
  └─→ 训练结束
```

### 6.3 评估流程详解

#### 6.3.1 EA 种群评估

```python
def evaluate_population(self, runner, t_env):
    for i, genome in enumerate(self.population):
        # 1. 重置适应度
        genome.reset_fitness()
        
        # 2. 评估个体
        avg_fitness = self.evaluate_genome(genome, runner, t_env)
        #   内部流程：
        #   - 保存原始 alloc_policy
        #   - 替换为 genome.alloc_policy
        #   - 运行 evaluation_episodes 个评估回合
        #   - 累加奖励到 genome.fitness
        #   - 恢复原始 alloc_policy
        
        # 3. 更新适应度历史
        genome.update_fitness_history()
        
        print(f"Genome {i}: avg_fitness = {avg_fitness:.4f}")
    
    # 4. 更新种群统计
    self._update_stats()
    
    # 5. 更新最优个体
    best_genome = max(self.population, key=lambda x: x.get_average_fitness())
    if best_genome.get_average_fitness() > self.best_genome.get_average_fitness():
        self.best_genome = best_genome.clone()
```

**时间复杂度**：
- 种群大小：5
- 评估回合数：5
- 总评估回合：5 × 5 = 25 个 episode
- **串行执行**：总时间 = 25 × episode_time

#### 6.3.2 主网络评估

```python
def evaluate_main_network_performance(self, runner, t_env):
    original_alloc_policy = runner.mac.alloc_policy  # 保存原始策略
    
    try:
        total_reward = 0.0
        for _ in range(self.evaluation_episodes):
            runner.run(test_mode=True)  # 运行一个 episode
            episode_reward = runner.last_episode_reward
            total_reward += episode_reward
        
        avg_fitness = total_reward / self.evaluation_episodes
        self.main_network_fitness_history.append(avg_fitness)
        return avg_fitness
    finally:
        runner.mac.alloc_policy = original_alloc_policy  # 恢复原始策略
```

---

## 配置参数详解

### 7.1 EA 配置参数

```yaml
ea_config:
  population_size: 5              # 种群大小：5 个个体
  elite_size: 2                    # 精英个体数量：保留 2 个最优个体
  mutation_rate: 0.1               # 变异率：10% 的参数发生变异
  mutation_strength: 0.1           # 变异强度：变异噪声的标准差
  crossover_rate: 0.8              # 交叉率：80% 的个体参与交叉
  selection_type: "tournament"     # 选择策略：锦标赛选择
  tournament_size: 3               # 锦标赛大小：每次选择 3 个个体竞争
  evaluation_episodes: 5           # 评估 episode 数：每个个体评估 5 个 episode
  sync_interval: 10000             # 同步间隔：每 10000 步进行一次 EA 评估和进化
  enable_bidirectional_sync: True  # 启用双向同步
  sync_threshold: 0.05             # 同步阈值：性能提升超过 0.05 才进行双向同步（RL→EA 和 EA→RL）
```

### 7.2 训练配置参数

```yaml
runner: "parallel"                 # 运行器类型：并行训练
batch_size_run: 8                  # 并行运行的批次大小：8 个环境同时运行
training_iters: 8                  # 每次训练迭代的步数：8 步
buffer_size: 5000                  # 经验回放缓冲区大小：存储 5000 个经验
target_update_interval: 150        # 目标网络更新间隔：每 150 个 episode 更新一次
```

### 7.3 网络配置参数

```yaml
agent:
  recurrent: False                 # 是否使用循环神经网络
  entity_scheme: True              # 是否使用实体方案
  subtask_cond: "mask"             # 子任务条件类型：使用掩码机制

hier_agent:
  task_allocation: "aql"           # 任务分配策略：注意力查询语言(AQL)
  n_proposals: 32                  # 提案数量：生成 32 个任务分配提案
  action_length: 5                 # 动作序列长度：每个动作序列长度为 5
  subtask_mask: True               # 是否使用子任务掩码
```

### 7.4 超参数选择建议

#### 7.4.1 种群大小

- **小种群（3-5）**：适合快速实验，计算开销小
- **中等种群（5-10）**：平衡探索和计算开销（推荐）
- **大种群（10+）**：适合复杂任务，但计算开销大

#### 7.4.2 同步间隔

- **短间隔（5000-10000）**：频繁评估，及时同步（推荐）
- **长间隔（20000+）**：减少计算开销，但可能错过重要改进

#### 7.4.3 评估回合数

- **少回合（3-5）**：快速评估，但可能不够稳定（推荐）
- **多回合（10+）**：更稳定的评估，但计算开销大

---

## 关键技术细节

### 8.1 串行评估实现

#### 8.1.1 为什么使用串行评估？

1. **避免嵌套多进程问题**：
   - 主训练使用 `ParallelRunner`（batch_size=8），已经创建了子进程
   - 如果 EA 评估也使用多进程，会导致嵌套多进程
   - CUDA 上下文在嵌套多进程中可能无法正常工作

2. **简化实现**：
   - 串行评估实现简单，易于调试
   - 避免进程间通信的复杂性

3. **资源管理**：
   - 避免创建过多进程导致 CPU/GPU 过载
   - 更好的资源利用率

#### 8.1.2 串行评估流程

```python
def evaluate_population(self, runner, t_env):
    # 串行循环评估每个个体
    for i, genome in enumerate(self.population):
        genome.reset_fitness()
        avg_fitness = self.evaluate_genome(genome, runner, t_env)
        # evaluate_genome 内部：
        #   1. 保存原始 alloc_policy
        #   2. 替换为 genome.alloc_policy
        #   3. 运行 evaluation_episodes 个评估回合
        #   4. 恢复原始 alloc_policy
        genome.update_fitness_history()
```

### 8.2 参数同步机制

#### 8.2.1 EA → RL 同步

```python
def sync_best_to_main(self, main_alloc_policy):
    if self.best_genome is not None:
        # 复制最优个体的参数到主网络
        for name, param in main_alloc_policy.named_parameters():
            if name in self.best_genome.parameters:
                param.data.copy_(self.best_genome.parameters[name])
```

**关键点**：
- 使用 `param.data.copy_()` 直接复制参数数据
- 只复制参数名匹配的参数
- 不复制优化器状态

#### 8.2.2 RL → EA 同步

```python
def _bidirectional_sync(self, main_network_fitness, main_alloc_policy):
    # 获取 EA 种群性能
    worst_ea_fitness = min([genome.get_average_fitness() for genome in self.population])
    
    # 阈值控制 RL → EA 同步
    if main_network_fitness > worst_ea_fitness + self.sync_threshold:
        # 用主网络替换 EA 最差个体
        self._replace_worst_with_main_network(main_network_fitness, main_alloc_policy)
        self.sync_stats['main_to_ea_syncs'] += 1
    else:
        self.sync_stats['no_main_to_ea_sync'] += 1
```

**关键点**：
- 只有当主网络明显优于 EA 最差个体时才替换
- 使用相同的 `sync_threshold` 阈值
- 创建新的 `AllocationGenome` 实例（深拷贝）
- 设置主网络的适应度值
- 直接替换最差个体

### 8.3 适应度计算

#### 8.3.1 个体适应度

```python
def evaluate_genome(self, genome, runner, t_env):
    total_reward = 0.0
    
    # 替换 alloc_policy
    original_alloc_policy = runner.mac.alloc_policy
    runner.mac.alloc_policy = genome.alloc_policy
    
    try:
        for _ in range(self.evaluation_episodes):
            runner.run(test_mode=True)
            episode_reward = runner.last_episode_reward
            total_reward += episode_reward
            genome.add_fitness(episode_reward)  # 累加奖励
    finally:
        runner.mac.alloc_policy = original_alloc_policy
    
    return total_reward / self.evaluation_episodes  # 返回平均适应度
```

**适应度 = 平均 episode 奖励**

#### 8.3.2 主网络适应度

```python
def evaluate_main_network_performance(self, runner, t_env):
    total_reward = 0.0
    for _ in range(self.evaluation_episodes):
        runner.run(test_mode=True)
        episode_reward = runner.last_episode_reward
        total_reward += episode_reward
    
    avg_fitness = total_reward / self.evaluation_episodes
    self.main_network_fitness_history.append(avg_fitness)
    return avg_fitness
```

### 8.4 精英保留策略

```python
def evolve_generation(self, main_network_fitness, main_alloc_policy):
    # 1. 双向同步（如果启用）
    if self.enable_bidirectional_sync:
        self._bidirectional_sync(main_network_fitness, main_alloc_policy)
    
    # 2. 选择父代
    num_parents = self.population_size - self.elite_size
    parents = self.selection.select(self.population, num_parents)
    
    # 3. 精英保留
    elite = sorted(self.population, key=lambda x: x.get_average_fitness(), reverse=True)[:self.elite_size]
    
    # 4. 生成新个体
    new_population = []
    
    # 4.1 保留精英
    for elite_genome in elite:
        new_population.append(elite_genome.clone())
    
    # 4.2 交叉和变异
    for i in range(0, len(parents), 2):
        if i + 1 < len(parents):
            parent1, parent2 = parents[i], parents[i + 1]
            child1, child2 = self.crossover.crossover(parent1, parent2)
            child1 = self.mutation.mutate(child1)
            child2 = self.mutation.mutate(child2)
            new_population.extend([child1, child2])
        else:
            child = self.mutation.mutate(parents[i])
            new_population.append(child)
    
    # 5. 更新种群
    self.population = new_population[:self.population_size]
    self.generation += 1
```

**关键点**：
- 精英数量：`elite_size = 2`（默认）
- 父代数量：`population_size - elite_size = 3`
- 精英直接保留到下一代，不参与交叉和变异

---

## 性能优化

### 9.1 计算开销分析

#### 9.1.1 EA 评估开销

假设：
- 种群大小：5
- 评估回合数：5
- Episode 平均时长：50 秒

**总评估时间**：
- 串行评估：5 × 5 × 50 = 1250 秒 ≈ 20.8 分钟
- 每 `sync_interval = 10000` 步执行一次

**相对训练时间**：
- 如果训练步长为 0.1 秒/步，10000 步 = 1000 秒 ≈ 16.7 分钟
- EA 评估时间 ≈ 训练时间的 1.25 倍

#### 9.1.2 优化建议

1. **减少评估回合数**：
   - 从 5 减少到 3，可减少 40% 的评估时间
   - 但可能降低评估稳定性

2. **增加同步间隔**：
   - 从 10000 增加到 20000，可减少 50% 的评估频率
   - 但可能错过重要改进

3. **减少种群大小**：
   - 从 5 减少到 3，可减少 40% 的评估时间
   - 但可能降低探索能力

### 9.2 内存管理

#### 9.2.1 基因组内存

每个 `AllocationGenome` 包含：
- `alloc_policy`: 完整的网络副本（深拷贝）
- `parameters`: 参数字典（参数数据的副本）

**内存估算**：
- 假设网络参数为 1MB
- 种群大小：5
- 总内存：5 × 1MB = 5MB（可忽略）

#### 9.2.2 评估时内存

评估时：
- 临时替换 `alloc_policy`（引用替换，不增加内存）
- 运行评估回合（正常内存使用）

**无额外内存开销**

### 9.3 GPU 使用

#### 9.3.1 评估时 GPU

- EA 评估时，网络在 GPU 上运行
- 每个个体评估时，网络参数在 GPU 上
- 无额外的 GPU 内存分配

#### 9.3.2 参数同步 GPU

- 参数同步使用 `param.data.copy_()`
- 如果参数在 GPU 上，同步也在 GPU 上
- 无 CPU-GPU 数据传输开销

---

## 使用指南

### 10.1 基本使用

#### 10.1.1 运行训练

```bash
cd /home/gud/ALMA-EA/src
python main.py \
    --env-config=ff \
    --config=qmix_atten_ea \
    --agent.subtask_cond=mask \
    --hier_agent.task_allocation=aql \
    --epsilon_anneal_time=2000000 \
    --use_tensorboard=True \
    --save_model=True \
    --save_model_interval=1000000 \
    --hier_agent.action_length=5
```

#### 10.1.2 配置文件

使用 `src/config/algs/qmix_atten_ea.yaml` 配置文件，其中包含：
- EA 配置参数
- 训练超参数
- 网络结构参数

### 10.2 参数调整

#### 10.2.1 调整 EA 参数

在配置文件中修改 `ea_config` 部分：

```yaml
ea_config:
  population_size: 5        # 调整种群大小
  elite_size: 2             # 调整精英数量
  mutation_rate: 0.1        # 调整变异率
  mutation_strength: 0.1    # 调整变异强度
  crossover_rate: 0.8       # 调整交叉率
  evaluation_episodes: 5    # 调整评估回合数
  sync_interval: 10000      # 调整同步间隔
  sync_threshold: 0.05      # 调整同步阈值
```

#### 10.2.2 调整选择策略

```yaml
ea_config:
  selection_type: "tournament"  # 可选: "tournament", "elite", "roulette"
  tournament_size: 3            # 仅当 selection_type="tournament" 时有效
```

#### 10.2.3 调整交叉类型

在代码中修改 `CrossoverOperator` 的 `crossover_type` 参数：
- `"uniform"`: 均匀交叉（默认）
- `"arithmetic"`: 算术交叉

### 10.3 监控和调试

#### 10.3.1 日志输出

EA 评估时会输出：
```
Evaluating population at generation 0...
Genome 0: avg_fitness = 0.5234
Genome 1: avg_fitness = 0.5123
Genome 2: avg_fitness = 0.4987
Genome 3: avg_fitness = 0.5102
Genome 4: avg_fitness = 0.5056
New best genome found with fitness: 0.5234
```

双向同步时会输出：
```
Bidirectional sync comparison:
  EA best: 0.5234, EA avg: 0.5100, EA worst: 0.4987
  Main network: 0.5200
  → Main network significantly outperforms worst EA: 0.5200 > 0.4987 + 0.0500
  → EA best (0.5234) not significantly better than RL (0.5200), no EA→RL sync
```

或者当主网络未达到阈值时：
```
Bidirectional sync comparison:
  EA best: 0.5234, EA avg: 0.5100, EA worst: 0.5050
  Main network: 0.5200
  → Main network (0.5200) not significantly better than worst EA (0.5050), no RL→EA sync
  → EA best (0.5234) not significantly better than RL (0.5200), no EA→RL sync
```

当 EA 最优达到阈值时：
```
Bidirectional sync comparison:
  EA best: 0.5800, EA avg: 0.5100, EA worst: 0.4987
  Main network: 0.5200
  → Main network significantly outperforms worst EA: 0.5200 > 0.4987 + 0.0500
  → EA best significantly outperforms main network: 0.5800 > 0.5200 + 0.0500
```

#### 10.3.2 统计信息

调用 `ea_manager.print_stats()` 可打印：
- 当前代数
- 最佳适应度
- 平均适应度
- 最差适应度
- 种群多样性（标准差）
- 同步统计信息

### 10.4 常见问题

#### 10.4.1 EA 评估时间过长

**原因**：
- 种群大小过大
- 评估回合数过多
- 同步间隔过短

**解决方案**：
- 减少 `population_size`（如从 5 到 3）
- 减少 `evaluation_episodes`（如从 5 到 3）
- 增加 `sync_interval`（如从 10000 到 20000）

#### 10.4.2 EA 种群性能不提升

**原因**：
- 变异率/强度过小
- 交叉率过小
- 选择压力不足

**解决方案**：
- 增加 `mutation_rate`（如从 0.1 到 0.2）
- 增加 `mutation_strength`（如从 0.1 到 0.15）
- 增加 `crossover_rate`（如从 0.8 到 0.9）
- 减少 `tournament_size`（增加选择压力）

#### 10.4.3 同步过于频繁/稀少

**原因**：
- 阈值设置不当

**解决方案**：
- 如果同步过于频繁，增加 `sync_threshold`（如从 0.05 到 0.1）
- 如果同步过于稀少，减少 `sync_threshold`（如从 0.05 到 0.02）

---

## 总结

ALMA-EA 方案通过结合强化学习和进化算法，实现了一种混合优化方法。核心特点包括：

1. **串行评估**：避免嵌套多进程问题，简化实现
2. **阈值控制双向同步**：EA 和 RL 之间的同步都通过阈值控制，确保稳定性
   - **RL→EA 同步**：只有当主网络明显优于 EA 最差个体时（超过阈值），才替换 EA 最差个体
   - **EA→RL 同步**：只有当 EA 最优个体明显优于主网络时（超过阈值），才同步到主网络
3. **精英保留**：每代保留最优个体，避免性能退化
4. **灵活配置**：支持多种选择、交叉、变异策略

该方案在保持 ALMA 框架原有功能的基础上，通过 EA 增强了任务分配策略的探索能力，实现了更好的性能。双向同步的阈值控制机制确保了知识交换的稳定性和有效性。

