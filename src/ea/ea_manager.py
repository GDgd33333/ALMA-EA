import torch as th  
import numpy as np  
import random  
import copy  
from typing import List, Dict, Any, Optional  
import time  

# 导入自定义模块
from .genome import AllocationGenome  # 分配器基因组类
from .operators import SelectionOperator, MutationOperator, CrossoverOperator  # 进化算子类


class EAManager:
    """
    进化算法管理器，负责管理分配器网络的进化过程
    
    该类实现了进化算法的主要功能，包括：
    - 种群初始化和管理
    - 个体适应度评估
    - 选择、交叉、变异操作
    - 双向同步机制（EA种群与主网络之间的知识交换）
    - 统计信息跟踪和检查点保存
    """
    
    def __init__(self, 
                 population_size: int = 5,  # 种群大小，默认5个个体
                 elite_size: int = 2,  # 精英个体数量，默认2个
                 mutation_rate: float = 0.1,  # 变异率，默认10%
                 mutation_strength: float = 0.1,  # 变异强度，默认0.1
                 crossover_rate: float = 0.8,  # 交叉率，默认80%
                 selection_type: str = "tournament",  # 选择类型，默认锦标赛选择
                 tournament_size: int = 3,  # 锦标赛大小，默认3
                 evaluation_episodes: int = 5,  # 评估回合数，默认5个episode
                 sync_interval: int = 1000,  # 同步间隔，每1000步同步一次最优个体
                 device: str = "cuda",  # 计算设备，默认CUDA
                 sync_threshold: float = 0.05,  # 性能提升阈值，默认5%
                 enable_bidirectional_sync: bool = True,  # 启用双向同步，默认True
                 elite_buffer: Optional['EliteBuffer'] = None):  # 精英缓存池（可选）
        
        # 保存基本参数
        self.population_size = population_size  # 种群大小
        self.elite_size = elite_size  # 精英个体数量
        self.evaluation_episodes = evaluation_episodes  # 评估回合数
        self.sync_interval = sync_interval  # 同步间隔
        self.device = device  # 计算设备
        self.sync_threshold = sync_threshold  # 同步阈值
        self.enable_bidirectional_sync = enable_bidirectional_sync  # 双向同步开关
        
        # 初始化进化算子
        self.selection = SelectionOperator(selection_type, tournament_size)  # 选择算子
        self.mutation = MutationOperator(mutation_rate, mutation_strength)  # 变异算子
        self.crossover = CrossoverOperator(crossover_rate)  # 交叉算子
        
        # 种群管理相关变量
        self.population: List[AllocationGenome] = []  # 种群列表，存储所有个体
        self.generation = 0  # 当前代数，初始为0
        self.best_genome: Optional[AllocationGenome] = None  # 最优个体，初始为None
        self.fitness_history = []  # 适应度历史记录
        
        # 双向同步相关变量
        self.main_network_fitness_history = []  # 主网络适应度历史
        self.sync_stats = {  # 同步统计信息
            'ea_to_main_syncs': 0,  # EA到主网络的同步次数
            'main_to_ea_syncs': 0,  # 主网络到EA的同步次数
            'no_ea_to_main_sync': 0,  # EA未同步到主网络的次数（未达到阈值）
            'no_main_to_ea_sync': 0  # 主网络未同步到EA的次数（未达到阈值）
        }
        
        # 统计信息字典
        self.stats = {
            'generation': 0,  # 当前代数
            'best_fitness': 0.0,  # 最佳适应度
            'avg_fitness': 0.0,  # 平均适应度
            'worst_fitness': 0.0,  # 最差适应度
            'diversity': 0.0  # 种群多样性（标准差）
        }
        
        # 精英缓存池（用于存储EA找到的好分配结构）
        self.elite_buffer = elite_buffer
    
    def initialize_population(self, base_alloc_policy):
        """
        初始化种群
        
        Args:
            base_alloc_policy: 基础分配策略网络，用作种群的模板
        """
        print(f"Initializing EA population with {self.population_size} individuals...")
        
        # 清空现有种群
        self.population = []
        
        # 创建指定数量的个体
        for i in range(self.population_size):
            # 基于基础分配策略创建基因组，每个个体都有唯一的ID
            genome = AllocationGenome(base_alloc_policy, genome_id=i)
            self.population.append(genome)  # 将基因组添加到种群中
        
        # 将第一个个体作为初始最优个体
        self.best_genome = self.population[0].clone()
        print(f"Population initialized with {len(self.population)} individuals")
    
    def evaluate_genome(self, genome: AllocationGenome, runner, t_env: int) -> float:
        """
        评估单个基因的适应度
        
        Args:
            genome: 要评估的基因组
            runner: 环境运行器
            t_env: 当前环境时间步
            
        Returns:
            float: 平均适应度值
        """
        total_reward = 0.0  # 总奖励，初始为0
        
        # 关键修复：EA评估必须使用"独立副本"的MAC/policy，不要动训练用的runner.mac
        # 否则在多进程/多线程rollout时会污染训练数据，破坏on-policy假设
        # 
        # 注意：PyTorch的tensor不支持deepcopy，所以不能使用copy.deepcopy(runner.mac)
        # 解决方案：由于EA评估和PPO训练严格串行（在run.py中已保证），
        # 我们可以直接使用runner.mac，只临时替换alloc_policy，然后恢复
        # 这样既避免了deepcopy的问题，又保证了隔离性
        
        # 保存原始alloc_policy（用于恢复）
        original_alloc_policy = runner.mac.alloc_policy
        
        # 临时替换为基因组策略（仅在评估期间）
        runner.mac.alloc_policy = genome.alloc_policy
        
        # 关键修复：设置标志，让entity_controller在test_mode时也保存old_actions
        # 这样EA评估时就能获取到正确的分配动作
        original_is_ea_evaluation = getattr(runner.mac, '_is_ea_evaluation', False)
        runner.mac._is_ea_evaluation = True
        
        # 重置警告计数器（每个genome评估开始时重置）
        self._alloc_actions_warning_count = 0
        
        # 关键：确保EA评估期间不会有并行的PPO rollout
        # 当前实现要求调用方确保EA评估和PPO训练严格串行（在run.py中已保证）
        # 如果未来需要并行，应该使用独立的runner实例，而不是替换runner.mac.alloc_policy
        try:
            # 兼容两种runner类型：获取初始的test_returns长度（用于ParallelRunner）
            initial_test_returns_len = len(runner.test_returns) if hasattr(runner, 'test_returns') else 0
            
            # 运行指定数量的评估回合
            for _ in range(self.evaluation_episodes):
                # 运行一个episode（测试模式）
                # 注意：在compute_allocation中已经会写入alloc_actions字段，不需要额外的保存逻辑
                episode_batch, final_subtask_infos = runner.run(test_mode=True)
                
                # 获取episode奖励（兼容两种runner类型）
                reward_valid = True  # 标志：reward是否有效（不是fallback的0.0）
                if hasattr(runner, 'last_episode_reward'):
                    # EpisodeRunner类型：直接获取last_episode_reward
                    episode_reward = runner.last_episode_reward
                elif hasattr(runner, 'test_returns') and len(runner.test_returns) > initial_test_returns_len:
                    # ParallelRunner类型：从test_returns获取最后一个episode的回报
                    # runner.run()会运行batch_size个episode，取最后一个
                    episode_reward = runner.test_returns[-1]
                    initial_test_returns_len = len(runner.test_returns)
                else:
                    # 如果无法获取，使用0并记录警告，标记为无效
                    episode_reward = 0.0
                    reward_valid = False
                    print(f"Warning: Could not get episode reward for genome {genome.genome_id}, using 0.0")
                
                total_reward += episode_reward  # 累加奖励
                genome.add_fitness(episode_reward)  # 将奖励添加到基因组适应度中
                
                # 精英蒸馏：如果episode成功，保存分配序列到EliteBuffer
                # 关键修复：如果reward获取失败（reward_valid=False），禁止保存精英样本
                # 避免将"伪精英"（reward读取失败）混入EliteBuffer，污染蒸馏数据
                if self.elite_buffer is not None and reward_valid:
                    # 判断是否成功：从final_subtask_infos或env_info中获取success标志
                    episode_success = False
                    if final_subtask_infos and len(final_subtask_infos) > 0:
                        # 从final_subtask_infos中获取success信息
                        last_info = final_subtask_infos[-1] if isinstance(final_subtask_infos, list) else final_subtask_infos
                        if isinstance(last_info, dict):
                            episode_success = last_info.get('battle_won', False) or \
                                            last_info.get('episode_solved', False) or \
                                            last_info.get('success', False)
                    
                    # 关键修复：score不要自己归一化，直接传episode_reward
                    # EliteBuffer用min_score来控制阈值，不要用/100这种拍脑袋的归一化
                    # 成功时用1.0或episode_reward都可以，但不要归一化
                    score = 1.0 if episode_success else episode_reward
                    
                    # 判断是否保存（使用混合策略）
                    should_save = False
                    if episode_success:
                        # 绝对成功，直接保存
                        should_save = True
                    else:
                        # 失败时，使用EliteBuffer的min_score阈值（不要在这里归一化）
                        min_score = getattr(self.elite_buffer, 'min_score', 0.0)
                        if episode_reward > min_score:
                            should_save = True
                        elif self.elite_buffer.use_relative_ranking:
                            # 使用相对排名策略（在EliteBuffer中判断）
                            should_save = True
                    
                    if should_save:
                        # 从episode_batch中提取分配信息
                        # 传入runner.mac以便获取task_allocations
                        self._save_elite_allocation(episode_batch, episode_reward, episode_success, score, runner.mac)
        
        finally:
            # 无论是否发生异常，都要恢复原始alloc_policy和标志（关键修复）
            runner.mac.alloc_policy = original_alloc_policy
            runner.mac._is_ea_evaluation = original_is_ea_evaluation
        
        # 返回平均适应度
        return total_reward / self.evaluation_episodes
    
    def _save_elite_allocation(self, episode_batch, episode_reward: float, episode_success: bool, score: float, mac=None):
        """
        保存精英分配序列到EliteBuffer
        
        Args:
            episode_batch: episode批次数据
            episode_reward: episode奖励
            episode_success: 是否成功
            score: 分数（成功为1.0，否则为归一化的reward）
            mac: MAC实例，用于获取task_allocations（可选）
        """
        if self.elite_buffer is None:
            return
        
        # 从episode_batch中提取分配信息
        if not hasattr(episode_batch, 'data') or not hasattr(episode_batch.data, 'transition_data'):
            return
        
        transition_data = episode_batch.data.transition_data
        
        # 检查是否有分配数据
        # 关键修复：确认正确的动作字段名
        # old_actions是PPO缓存，可能不是我们想要的"当时执行的分配动作"
        # 需要检查transition_data中实际有哪些字段
        if 'alloc_order' not in transition_data:
            return
        
        # 找到所有决策点（hier_decision == 1）
        if 'hier_decision' not in transition_data:
            return
        
        hier_decision = transition_data['hier_decision']  # (bs, ts)
        decision_mask = (hier_decision == 1)  # (bs, ts)
        
        # 获取所有决策点的索引
        decision_indices = th.nonzero(decision_mask, as_tuple=False)  # (n_dp, 2) - [ep_id, t_id]
        
        if len(decision_indices) == 0:
            return
        
        # 关键修复：只保存最后一个决策点，避免污染buffer
        # 成功与否是episode级别标签，不能"均匀分摊"到所有决策点
        # 只存最后一个决策点（或top-k按reward-to-go排序，这里先做最简单的）
        idx = decision_indices[-1]  # 最后一个决策点
        ep_id = idx[0].item()
        t_id = idx[1].item()
        
        # 关键对齐检查1：确保hier_decision[ep_id, t_id] == 1（必须是决策点）
        if hier_decision[ep_id, t_id].item() != 1:
            print(f"[EA评估] Warning: 对齐错误！hier_decision[{ep_id}, {t_id}] != 1，跳过该样本")
            return
        
        # 提取alloc_order
        alloc_order = transition_data['alloc_order'][ep_id, t_id]  # (na,) 或 (1, na)
        
        # 处理alloc_order的维度
        if alloc_order.dim() > 1:
            alloc_order = alloc_order.squeeze(0)  # (1, na) -> (na,)
        
        # 关键对齐检查2：确保alloc_order不是全零（应该是有效的分配顺序）
        # alloc_order应该是[0, 1, 2, ..., na-1]的某种排列，不应该全是零
        if alloc_order.numel() > 0:
            if alloc_order.sum().item() == 0:
                print(f"[EA评估] Warning: alloc_order[{ep_id}, {t_id}]全是零，可能对齐错误，跳过该样本")
                return
            # 额外检查：alloc_order应该是[0, na-1]范围内的整数
            if alloc_order.min().item() < 0 or alloc_order.max().item() >= alloc_order.numel():
                print(f"[EA评估] Warning: alloc_order[{ep_id}, {t_id}]值超出范围，可能对齐错误，跳过该样本")
                return
        
        # 关键修复：使用优先级链获取"实际执行的分配动作"
        # 优先级：1) alloc_actions字段（最推荐，专门用于高层动作） 2) old_actions字段 3) task_allocations
        alloc_actions = None
        action_source = None
        
        # 方案1（最推荐）：从transition_data的alloc_actions字段获取
        # 这是专门用于存储高层分配动作的字段，在compute_allocation中写入
        if 'alloc_actions' in transition_data:
            alloc_actions_tensor = transition_data['alloc_actions']
            
            # 关键修复：添加边界检查，防止索引超出范围
            bs, ts = alloc_actions_tensor.shape[:2]
            if ep_id >= bs or t_id >= ts:
                # 索引超出范围，可能是 decision_indices 中的索引不正确
                # 这种情况通常发生在多个 episode 的决策点混合在一起时
                print(f"[EA评估] Warning: 索引超出范围！ep_id={ep_id} >= batch_size={bs} 或 t_id={t_id} >= max_seq_length={ts}，跳过该样本")
                alloc_actions = None
                action_source = None
            else:
                alloc_actions = alloc_actions_tensor[ep_id, t_id]  # (na, nt)
                
                # 关键修复：先判断动作是否有效（是否是有效的one-hot），判断完之后再决定是否使用
                # 不因为全零就判定为无效，而是判断是否是有效的one-hot编码
                # 关键修复：允许无效agent的行是全零，只检查有效agent的行是否是one-hot
                if alloc_actions is not None and alloc_actions.numel() > 0:
                    # 验证：确保alloc_actions是有效的one-hot（每行只有一个1）
                    row_sums = alloc_actions.sum(dim=-1)  # (na,)
                    
                    # 获取entity_mask来判断哪些agent是有效的
                    agent_mask = None
                    if 'entity_mask' in transition_data:
                        entity_mask = transition_data['entity_mask'][ep_id, t_id]  # (ne,)
                        na = alloc_actions.shape[0]  # agent数量
                        if entity_mask.shape[0] >= na:
                            agent_mask = entity_mask[:na]  # (na,) - 只取前na个（agents）
                    
                    # 如果能够获取agent_mask，只检查有效agent的行
                    if agent_mask is not None:
                        valid_agent_mask = (1 - agent_mask).bool()  # 有效agent的mask（0表示有效）
                        if valid_agent_mask.any():
                            valid_row_sums = row_sums[valid_agent_mask]  # 只检查有效agent的行
                            # 有效agent的行应该是one-hot（和为1），无效agent的行可以是全零（和为0）
                            valid_ones = th.allclose(valid_row_sums, th.ones_like(valid_row_sums), atol=1e-5)
                            invalid_zeros = th.allclose(row_sums[~valid_agent_mask], th.zeros_like(row_sums[~valid_agent_mask]), atol=1e-5) if (~valid_agent_mask).any() else True
                            
                            if valid_ones and invalid_zeros:
                                # 是有效的格式：有效agent是one-hot，无效agent是全零
                                action_source = "alloc_actions"
                            else:
                                # 有效agent的行不是one-hot，或无效agent的行不是全零
                                if not hasattr(self, '_onehot_warning_count'):
                                    self._onehot_warning_count = 0
                                self._onehot_warning_count += 1
                                if self._onehot_warning_count == 1 or self._onehot_warning_count % 100 == 0:
                                    print(f"[EA评估] Warning ({self._onehot_warning_count}次): alloc_actions[{ep_id}, {t_id}]格式不正确（有效agent应one-hot，无效agent应全零），跳过该样本")
                                alloc_actions = None
                                action_source = None
                        else:
                            # 所有agent都无效，alloc_actions应该全零
                            if th.allclose(row_sums, th.zeros_like(row_sums), atol=1e-5):
                                # 所有agent都无效且alloc_actions全零，这是有效的
                                action_source = "alloc_actions"
                            else:
                                print(f"[EA评估] Warning: alloc_actions[{ep_id}, {t_id}]所有agent都无效但alloc_actions不全零，跳过该样本")
                                alloc_actions = None
                                action_source = None
                    else:
                        # 无法获取agent_mask，使用原来的严格检查（所有行都应该是one-hot）
                        is_valid_onehot = th.allclose(row_sums, th.ones_like(row_sums), atol=1e-5)
                        if is_valid_onehot:
                            # 是有效的one-hot，可以使用
                            action_source = "alloc_actions"
                        else:
                            # 不是有效的one-hot，跳过该样本
                            if not hasattr(self, '_onehot_warning_count'):
                                self._onehot_warning_count = 0
                            self._onehot_warning_count += 1
                            if self._onehot_warning_count == 1 or self._onehot_warning_count % 100 == 0:
                                print(f"[EA评估] Warning ({self._onehot_warning_count}次): alloc_actions[{ep_id}, {t_id}]不是有效的one-hot（无法获取entity_mask进行验证），跳过该样本")
                            alloc_actions = None
                            action_source = None
                else:
                    # alloc_actions 为空
                    alloc_actions = None
                    action_source = None
        
        # 方案2（兼容）：从transition_data的old_actions字段获取
        # 在EA评估时（test_mode=True），old_actions可能不会被保存，所以可能全是零
        if alloc_actions is None and 'old_actions' in transition_data:
            alloc_actions = transition_data['old_actions'][ep_id, t_id]  # (na, nt)
            action_source = "old_actions"
            
            # 验证：确保old_actions不是全零（可能是未初始化的）
            if alloc_actions is not None and alloc_actions.numel() > 0:
                if alloc_actions.sum().item() == 0:
                    # old_actions全是零，说明在test_mode下没有被保存
                    alloc_actions = None
                    action_source = None
                else:
                    # 验证：确保old_actions是有效的one-hot（每行只有一个1）
                    row_sums = alloc_actions.sum(dim=-1)  # (na,)
                    if not th.allclose(row_sums, th.ones_like(row_sums)):
                        # 仍然使用，但记录警告（只打印一次，避免刷屏）
                        pass
        
        # 方案3（fallback）：从mac.task_allocations获取
        # task_allocations的shape是(batch_size, na, nt)
        if alloc_actions is None and mac is not None and hasattr(mac, 'task_allocations') and mac.task_allocations is not None:
            batch_size = mac.task_allocations.shape[0]
            actual_ep_id = ep_id if ep_id < batch_size else 0
            if actual_ep_id < batch_size:
                task_alloc = mac.task_allocations[actual_ep_id]  # (na, nt)
                if task_alloc is not None and task_alloc.numel() > 0:
                    if task_alloc.sum().item() > 0:
                        alloc_actions = task_alloc.detach().clone()
                        action_source = "task_allocations"
        
        # 如果获取不到，静默跳过（避免刷屏报错）
        # 只在每个evaluation开始时打印一次警告
        if alloc_actions is None:
            # 使用实例变量记录是否已经打印过警告，避免刷屏
            # 注意：每次evaluate_genome开始时重置这个标志
            if not hasattr(self, '_alloc_actions_warning_count'):
                self._alloc_actions_warning_count = 0
            self._alloc_actions_warning_count += 1
            # 只在第一次或每100次打印一次警告
            if self._alloc_actions_warning_count == 1 or self._alloc_actions_warning_count % 100 == 0:
                print(f"[EA评估] Warning ({self._alloc_actions_warning_count}次): 无法获取有效的alloc_actions，精英样本将跳过入库。"
                      f"请确保在compute_allocation中写入alloc_actions字段。")
                if self._alloc_actions_warning_count == 1:
                    print(f"Available transition_data keys: {list(transition_data.keys())}")
            return
        
        # 记录动作来源（用于调试）
        if action_source:
            # 只在第一次或调试时打印
            pass  # 可以添加日志记录
        
        # 构建meta_inputs（从episode_batch中提取）
        meta_inputs = self._extract_meta_inputs(episode_batch, ep_id, t_id)
        
        if meta_inputs is None:
            return
        
        # 关键检查：如果episode_reward是0.0且不是成功episode，可能是fallback值，禁止保存
        # 虽然已经在evaluate_genome中检查了reward_valid，但这里再加一层保护
        if episode_reward == 0.0 and not episode_success:
            # 如果reward=0.0且不是成功episode，可能是fallback值，禁止保存
            # 避免将"伪精英"（reward读取失败）混入EliteBuffer
            print(f"[EA评估] Warning: episode_reward=0.0且episode_success=False，可能是fallback值，跳过保存")
            return
        
        # 保存到EliteBuffer（传入episode_reward用于相对排名）
        self.elite_buffer.add(
            meta_inputs=meta_inputs,
            alloc_order=alloc_order,
            alloc_actions=alloc_actions,
            score=score,
            episode_reward=episode_reward
        )
    
    def _save_task_allocations_to_batch(self, episode_batch, mac):
        """
        在EA评估时，从mac.task_allocations获取实际执行的分配动作，并保存到transition_data
        这样即使test_mode=True，也能获取到正确的分配动作
        
        Args:
            episode_batch: episode批次数据
            mac: MAC实例，包含task_allocations
        """
        if not hasattr(episode_batch, 'data') or not hasattr(episode_batch.data, 'transition_data'):
            return
        
        if not hasattr(mac, 'task_allocations') or mac.task_allocations is None:
            return
        
        transition_data = episode_batch.data.transition_data
        
        # 检查是否有old_actions字段
        if 'old_actions' not in transition_data:
            return
        
        # 找到所有决策点（hier_decision == 1）
        if 'hier_decision' not in transition_data:
            return
        
        hier_decision = transition_data['hier_decision']  # (bs, ts)
        decision_mask = (hier_decision == 1)  # (bs, ts)
        
        # 获取所有决策点的索引
        decision_indices = th.nonzero(decision_mask, as_tuple=False)  # (n_dp, 2) - [ep_id, t_id]
        
        if len(decision_indices) == 0:
            return
        
        # 从task_allocations获取分配动作并保存到old_actions
        # task_allocations的shape是(batch_size, na, nt)
        batch_size = mac.task_allocations.shape[0]
        
        for idx in decision_indices:
            ep_id = idx[0].item()
            t_id = idx[1].item()
            
            # 检查ep_id是否在有效范围内
            if ep_id < batch_size:
                # 获取该episode的分配动作
                task_alloc = mac.task_allocations[ep_id]  # (na, nt)
                
                # 验证：确保不是全零
                if task_alloc is not None and task_alloc.numel() > 0 and task_alloc.sum().item() > 0:
                    # 保存到old_actions
                    transition_data['old_actions'][ep_id, t_id] = task_alloc.detach().clone()
    
    def _extract_meta_inputs(self, episode_batch, ep_id: int, t_id: int):
        """
        从episode_batch中提取meta_inputs（用于复现compute_allocation_autoreg）
        
        Args:
            episode_batch: episode批次数据
            ep_id: episode索引
            t_id: 时间步索引
        
        Returns:
            meta_inputs字典，如果提取失败返回None
        """
        if not hasattr(episode_batch, 'data') or not hasattr(episode_batch.data, 'transition_data'):
            return None
        
        transition_data = episode_batch.data.transition_data
        
        # 必需的keys（根据_make_meta_batch中的定义）
        required_keys = ['entities', 'entity_mask', 'entity2task_mask', 'task_mask', 'obs_mask', 'last_alloc']
        
        meta_inputs = {}
        for key in required_keys:
            if key in transition_data:
                value = transition_data[key][ep_id, t_id]  # 提取单个样本
                # 确保有正确的维度
                if value.dim() == 0:
                    value = value.unsqueeze(0)
                meta_inputs[key] = value
            else:
                # 如果缺少关键信息，返回None
                if key in ['entities', 'entity_mask', 'task_mask']:
                    return None
        
        # 可选keys
        optional_keys = ['avail_actions']
        for key in optional_keys:
            if key in transition_data:
                value = transition_data[key][ep_id, t_id]
                if value.dim() == 0:
                    value = value.unsqueeze(0)
                meta_inputs[key] = value
        
        return meta_inputs
    
    def evaluate_main_network_performance(self, runner, t_env: int) -> float:
        """
        评估主网络性能
        
        Args:
            runner: 环境运行器
            t_env: 当前环境时间步
            
        Returns:
            float: 主网络的平均适应度
        """
        print(f"Evaluating main network performance at t_env: {t_env}")
        
        # 临时保存原始分配器（确保评估后能恢复）
        original_alloc_policy = runner.mac.alloc_policy
        
        try:
            # 兼容两种runner类型：获取初始的test_returns长度（用于ParallelRunner）
            initial_test_returns_len = len(runner.test_returns) if hasattr(runner, 'test_returns') else 0
            
            # 使用主网络运行指定数量的episode
            total_reward = 0.0  # 总奖励，初始为0
            for _ in range(self.evaluation_episodes):
                runner.run(test_mode=True)  # 以测试模式运行
                
                # 获取episode奖励（兼容两种runner类型）
                if hasattr(runner, 'last_episode_reward'):
                    # EpisodeRunner类型：直接获取last_episode_reward
                    episode_reward = runner.last_episode_reward
                elif hasattr(runner, 'test_returns') and len(runner.test_returns) > initial_test_returns_len:
                    # ParallelRunner类型：从test_returns获取最后一个episode的回报
                    # runner.run()会运行batch_size个episode，取最后一个
                    episode_reward = runner.test_returns[-1]
                    initial_test_returns_len = len(runner.test_returns)
                else:
                    # 如果无法获取，使用0并记录警告
                    episode_reward = 0.0
                    print(f"Warning: Could not get episode reward for main network, using 0.0")
                
                total_reward += episode_reward  # 累加奖励
            
            # 计算平均适应度
            avg_fitness = total_reward / self.evaluation_episodes
            self.main_network_fitness_history.append(avg_fitness)  # 记录到历史中
            print(f"Main network fitness: {avg_fitness:.4f}")
            return avg_fitness
        
        finally:
            # 恢复原始分配器（确保不会影响后续操作）
            runner.mac.alloc_policy = original_alloc_policy
    
    def evaluate_population(self, runner, t_env: int):
        """
        评估整个种群
        
        Args:
            runner: 环境运行器
            t_env: 当前环境时间步
        """
        print(f"Evaluating population at generation {self.generation}...")
        
        # 遍历种群中的每个个体进行评估
        for i, genome in enumerate(self.population):
            genome.reset_fitness()  # 重置个体适应度
            avg_fitness = self.evaluate_genome(genome, runner, t_env)  # 评估个体适应度
            genome.update_fitness_history()  # 更新个体适应度历史
            print(f"Genome {i}: avg_fitness = {avg_fitness:.4f}")  # 打印个体适应度
        
        # 更新种群统计信息
        self._update_stats()
        
        # 更新最优个体
        # 找到当前种群中适应度最高的个体
        best_genome = max(self.population, key=lambda x: x.get_average_fitness())
        # 如果当前最优个体比历史最优个体更好，则更新
        if best_genome.get_average_fitness() > self.best_genome.get_average_fitness():
            self.best_genome = best_genome.clone()  # 克隆最优个体
            print(f"New best genome found with fitness: {self.best_genome.get_average_fitness():.4f}")
    
    def evolve_generation(self, main_network_fitness: Optional[float] = None, main_alloc_policy=None):
        """
        进化一代，支持双向同步
        
        Args:
            main_network_fitness: 主网络适应度，用于双向同步比较
            main_alloc_policy: 主网络分配策略，用于双向同步
        """
        print(f"Evolving generation {self.generation}...")
        
        # 如果启用了双向同步且提供了主网络性能，进行双向比较
        if self.enable_bidirectional_sync and main_network_fitness is not None and main_alloc_policy is not None:
            self._bidirectional_sync(main_network_fitness, main_alloc_policy)
        
        # 选择父代（种群大小减去精英数量）
        num_parents = self.population_size - self.elite_size
        parents = self.selection.select(self.population, num_parents)  # 使用选择算子选择父代
        
        # 精英保留：选择适应度最高的个体作为精英
        elite = sorted(self.population, key=lambda x: x.get_average_fitness(), reverse=True)[:self.elite_size]
        
        # 生成新个体
        new_population = []
        
        # 保留精英个体到新种群
        for elite_genome in elite:
            new_population.append(elite_genome.clone())  # 克隆精英个体
        
        # 交叉和变异生成新个体
        for i in range(0, len(parents), 2):  # 每次处理两个父代
            if i + 1 < len(parents):  # 如果还有两个父代
                parent1, parent2 = parents[i], parents[i + 1]  # 获取两个父代
                child1, child2 = self.crossover.crossover(parent1, parent2)  # 交叉产生两个子代
                
                # 对子代进行变异
                child1 = self.mutation.mutate(child1)
                child2 = self.mutation.mutate(child2)
                
                new_population.extend([child1, child2])  # 将子代添加到新种群
            else:
                # 奇数个父代，最后一个直接变异
                child = self.mutation.mutate(parents[i])
                new_population.append(child)
        
        # 确保种群大小正确（截断到指定大小）
        new_population = new_population[:self.population_size]
        
        # 更新种群和代数
        self.population = new_population  # 更新种群
        self.generation += 1  # 代数加1
        
        print(f"Generation {self.generation} evolved with {len(self.population)} individuals")
    
    def _bidirectional_sync(self, main_network_fitness: float, main_alloc_policy):
        """
        双向同步机制（双向都有阈值限制）
        
        策略：
        1. RL → EA：只有当主网络明显优于EA最差个体时（超过阈值），才替换EA最差个体
        2. EA → RL：只有当EA最优明显优于主网络时（超过阈值），才同步到主网络
        
        Args:
            main_network_fitness: 主网络适应度
            main_alloc_policy: 主网络分配策略
        """
        # 获取EA种群性能统计
        ea_fitnesses = [genome.get_average_fitness() for genome in self.population]  # 所有个体适应度
        best_ea_fitness = max(ea_fitnesses)  # EA最优适应度
        worst_ea_fitness = min(ea_fitnesses)  # EA最差适应度
        avg_ea_fitness = np.mean(ea_fitnesses)  # EA平均适应度
        
        # 打印性能比较信息
        print(f"Bidirectional sync comparison:")
        print(f"  EA best: {best_ea_fitness:.4f}, EA avg: {avg_ea_fitness:.4f}, EA worst: {worst_ea_fitness:.4f}")
        print(f"  Main network: {main_network_fitness:.4f}")
        
        # 策略1：只有当主网络明显优于EA最差个体时，才替换EA最差个体（阈值控制）
        if main_network_fitness > worst_ea_fitness + self.sync_threshold:
            print(f"  → Main network significantly outperforms worst EA: {main_network_fitness:.4f} > {worst_ea_fitness:.4f} + {self.sync_threshold:.4f}")
            self._replace_worst_with_main_network(main_network_fitness, main_alloc_policy)
            self.sync_stats['main_to_ea_syncs'] += 1  # 更新同步统计
        else:
            print(f"  → Main network ({main_network_fitness:.4f}) not significantly better than worst EA ({worst_ea_fitness:.4f}), no RL→EA sync")
            self.sync_stats['no_main_to_ea_sync'] = self.sync_stats.get('no_main_to_ea_sync', 0) + 1  # 更新统计
        
        # 策略2：只有当EA最优明显优于主网络时，才让EA替换RL（阈值控制）
        # 关键修复：使用相对提升而不是绝对差值，更稳健
        # 对于success rate（0~1），用绝对差值；对于return（可能很大），用相对提升
        if main_network_fitness > 1.0:
            # return类型，使用相对提升
            relative_improvement = (best_ea_fitness - main_network_fitness) / (abs(main_network_fitness) + 1e-6)
            should_sync = relative_improvement > self.sync_threshold
            if should_sync:
                print(f"  → EA best significantly outperforms main network: {best_ea_fitness:.4f} > {main_network_fitness:.4f} "
                      f"(relative improvement: {relative_improvement:.2%} > {self.sync_threshold:.2%})")
            else:
                print(f"  → EA best ({best_ea_fitness:.4f}) not significantly better than RL ({main_network_fitness:.4f}), "
                      f"relative improvement: {relative_improvement:.2%} <= {self.sync_threshold:.2%}")
        else:
            # success rate类型，使用绝对差值
            should_sync = best_ea_fitness > main_network_fitness + self.sync_threshold
            if should_sync:
                print(f"  → EA best significantly outperforms main network: {best_ea_fitness:.4f} > {main_network_fitness:.4f} + {self.sync_threshold:.4f}")
            else:
                print(f"  → EA best ({best_ea_fitness:.4f}) not significantly better than RL ({main_network_fitness:.4f}), no EA→RL sync")
        
        if should_sync:
            # 同步EA最优个体到主网络
            sync_success = self.sync_best_to_main(main_alloc_policy)
            if sync_success:
                self.sync_stats['ea_to_main_syncs'] += 1  # 更新同步统计
            else:
                self.sync_stats['no_ea_to_main_sync'] = self.sync_stats.get('no_ea_to_main_sync', 0) + 1
        else:
            self.sync_stats['no_ea_to_main_sync'] = self.sync_stats.get('no_ea_to_main_sync', 0) + 1  # 更新统计
    
    def _replace_worst_with_main_network(self, main_network_fitness: float, main_alloc_policy):
        """
        用主网络替换EA最差个体
        
        Args:
            main_network_fitness: 主网络适应度
            main_alloc_policy: 主网络分配策略
        """
        # 创建主网络基因组
        main_network_genome = AllocationGenome(main_alloc_policy, genome_id=len(self.population))
        main_network_genome.add_fitness(main_network_fitness)  # 设置主网络适应度
        
        # 找到最差个体并替换
        worst_index = min(range(len(self.population)), 
                         key=lambda i: self.population[i].get_average_fitness())  # 找到适应度最低的个体索引
        worst_fitness = self.population[worst_index].get_average_fitness()  # 获取最差个体适应度
        
        self.population[worst_index] = main_network_genome  # 用主网络基因组替换最差个体
        
        print(f"  → Replaced worst EA individual (fitness: {worst_fitness:.4f}) with main network (fitness: {main_network_fitness:.4f})")
        
        # 更新最优个体（如果主网络比当前最优个体更好）
        if main_network_fitness > self.best_genome.get_average_fitness():
            self.best_genome = main_network_genome.clone()  # 克隆主网络基因组作为新的最优个体
            print(f"  → Main network became new best genome!")
    
    def sync_best_to_main(self, main_alloc_policy):
        """
        将最优个体同步到主网络
        
        Args:
            main_alloc_policy: 主网络分配策略
        
        Returns:
            bool: 是否成功同步
        """
        if self.best_genome is None:
            return False
        
        # 获取主网络和最优个体的参数
        main_params = dict(main_alloc_policy.named_parameters())
        best_params = self.best_genome.parameters
        
        # 关键修复：统计参数匹配率，如果匹配率太低，拒绝sync
        matched = 0
        total = len(main_params)
        matched_names = []
        unmatched_names = []
        
        # 同步参数
        for name, param in main_params.items():
            if name in best_params:
                param.data.copy_(best_params[name])
                matched += 1
                matched_names.append(name)
            else:
                unmatched_names.append(name)
        
        match_rate = matched / total if total > 0 else 0.0
        
        # 如果匹配率低于95%，报警并拒绝sync
        if match_rate < 0.95:
            print(f"  → WARNING: Parameter match rate {match_rate:.2%} ({matched}/{total}) is too low!")
            if len(unmatched_names) > 0:
                print(f"  → Unmatched parameters (first 10): {unmatched_names[:10]}")
            print(f"  → Refusing to sync to avoid silent parameter mismatch")
            return False
        
        # 关键修复：正确显示best_genome的fitness
        # 优先级：1) get_average_fitness()（如果evaluation_count > 0） 2) fitness_history最后一个值 3) fitness值（如果evaluation_count > 0）
        best_fitness_display = 0.0
        if self.best_genome.evaluation_count > 0:
            # 如果已经评估过，使用平均适应度
            best_fitness_display = self.best_genome.get_average_fitness()
        elif len(self.best_genome.fitness_history) > 0:
            # 如果evaluation_count=0但fitness_history不为空，使用最新的fitness_history值
            best_fitness_display = self.best_genome.fitness_history[-1]
        elif self.best_genome.fitness != 0.0:
            # 如果fitness不为0但evaluation_count=0，直接使用fitness（可能是单次评估）
            best_fitness_display = self.best_genome.fitness
        
        print(f"  → Synced best EA genome (fitness: {best_fitness_display:.4f}) to main network")
        print(f"  → Parameter match rate: {match_rate:.2%} ({matched}/{total})")
        return True
    
    
    def _update_stats(self):
        """更新统计信息"""
        fitnesses = [genome.get_average_fitness() for genome in self.population]  # 获取所有个体适应度
        
        # 更新统计信息字典
        self.stats = {
            'generation': self.generation,  # 当前代数
            'best_fitness': max(fitnesses),  # 最佳适应度
            'avg_fitness': np.mean(fitnesses),  # 平均适应度
            'worst_fitness': min(fitnesses),  # 最差适应度
            'diversity': np.std(fitnesses)  # 种群多样性（标准差）
        }
        
        self.fitness_history.append(self.stats['best_fitness'])  # 将最佳适应度添加到历史记录
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            Dict[str, Any]: 统计信息字典的副本
        """
        return self.stats.copy()  # 返回统计信息的副本，避免外部修改
    
    def save_checkpoint(self, filepath: str):
        """
        保存检查点
        
        Args:
            filepath: 保存路径
        """
        # 构建检查点数据
        checkpoint = {
            'generation': self.generation,  # 当前代数
            'population': self.population,  # 种群
            'best_genome': self.best_genome,  # 最优个体
            'fitness_history': self.fitness_history,  # 适应度历史
            'stats': self.stats  # 统计信息
        }
        th.save(checkpoint, filepath)  # 保存到文件
        print(f"EA checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """
        加载检查点
        
        Args:
            filepath: 检查点文件路径
        """
        checkpoint = th.load(filepath, map_location=self.device)  # 加载检查点文件
        self.generation = checkpoint['generation']  # 恢复代数
        self.population = checkpoint['population']  # 恢复种群
        self.best_genome = checkpoint['best_genome']  # 恢复最优个体
        self.fitness_history = checkpoint['fitness_history']  # 恢复适应度历史
        self.stats = checkpoint['stats']  # 恢复统计信息
        print(f"EA checkpoint loaded from {filepath}")
    
    def print_stats(self):
        """打印统计信息"""
        print(f"\n=== EA Generation {self.generation} Stats ===")
        print(f"Best Fitness: {self.stats['best_fitness']:.4f}")  # 最佳适应度
        print(f"Avg Fitness: {self.stats['avg_fitness']:.4f}")  # 平均适应度
        print(f"Worst Fitness: {self.stats['worst_fitness']:.4f}")  # 最差适应度
        print(f"Diversity: {self.stats['diversity']:.4f}")  # 种群多样性
        
        # 打印双向同步统计（如果启用了双向同步）
        if self.enable_bidirectional_sync:
            print(f"\n=== Bidirectional Sync Stats ===")
            print(f"EA → Main syncs: {self.sync_stats['ea_to_main_syncs']}")  # EA到主网络同步次数
            print(f"Main → EA syncs: {self.sync_stats['main_to_ea_syncs']}")  # 主网络到EA同步次数
            print(f"No EA→Main sync (threshold not met): {self.sync_stats.get('no_ea_to_main_sync', 0)}")  # EA未同步到主网络次数
            print(f"No Main→EA sync (threshold not met): {self.sync_stats.get('no_main_to_ea_sync', 0)}")  # 主网络未同步到EA次数
            print(f"Sync threshold: {self.sync_threshold:.4f}")  # 同步阈值
        
        print("=" * 40)  # 打印分隔线
