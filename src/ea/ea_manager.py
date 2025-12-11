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
                 enable_bidirectional_sync: bool = True):  # 启用双向同步，默认True
        
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
        
        # 临时替换主网络的分配器为当前基因组
        original_alloc_policy = runner.mac.alloc_policy  # 保存原始分配策略
        runner.mac.alloc_policy = genome.alloc_policy  # 替换为基因组策略
        
        try:
            # 兼容两种runner类型：获取初始的test_returns长度（用于ParallelRunner）
            initial_test_returns_len = len(runner.test_returns) if hasattr(runner, 'test_returns') else 0
            
            # 运行指定数量的评估回合
            for _ in range(self.evaluation_episodes):
                # 运行一个episode（测试模式）
                runner.run(test_mode=True)
                
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
                    print(f"Warning: Could not get episode reward for genome {genome.genome_id}, using 0.0")
                
                total_reward += episode_reward  # 累加奖励
                genome.add_fitness(episode_reward)  # 将奖励添加到基因组适应度中
        
        finally:
            # 无论是否发生异常，都要恢复原始分配器
            runner.mac.alloc_policy = original_alloc_policy
        
        # 返回平均适应度
        return total_reward / self.evaluation_episodes
    
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
        if best_ea_fitness > main_network_fitness + self.sync_threshold:
            print(f"  → EA best significantly outperforms main network: {best_ea_fitness:.4f} > {main_network_fitness:.4f} + {self.sync_threshold:.4f}")
            # 同步EA最优个体到主网络
            self.sync_best_to_main(main_alloc_policy)
            self.sync_stats['ea_to_main_syncs'] += 1  # 更新同步统计
        else:
            print(f"  → EA best ({best_ea_fitness:.4f}) not significantly better than RL ({main_network_fitness:.4f}), no EA→RL sync")
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
        """
        if self.best_genome is not None:  # 如果存在最优个体
            # 复制最优个体的参数到主网络
            for name, param in main_alloc_policy.named_parameters():  # 遍历主网络的所有参数
                if name in self.best_genome.parameters:  # 如果参数名在最优个体中存在
                    param.data.copy_(self.best_genome.parameters[name])  # 复制参数数据
            print(f"Synced best genome (fitness: {self.best_genome.get_average_fitness():.4f}) to main network")
    
    def should_sync(self, t_env: int) -> bool:
        """
        判断是否应该同步最优个体
        
        Args:
            t_env: 当前环境时间步
            
        Returns:
            bool: 是否应该同步
        """
        return t_env % self.sync_interval == 0 and t_env > 0  # 检查是否到了同步间隔且时间步大于0
    
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
