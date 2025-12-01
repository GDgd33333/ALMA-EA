import torch as th
import numpy as np
import random
from typing import List, Tuple
from .genome import AllocationGenome


class SelectionOperator:
    """选择算子"""
    
    def __init__(self, selection_type: str = "tournament", tournament_size: int = 3):
        self.selection_type = selection_type
        self.tournament_size = tournament_size
    
    def select(self, population: List[AllocationGenome], num_parents: int) -> List[AllocationGenome]:
        """选择父代个体"""
        if self.selection_type == "tournament":
            return self._tournament_selection(population, num_parents)
        elif self.selection_type == "elite":
            return self._elite_selection(population, num_parents)
        elif self.selection_type == "roulette":
            return self._roulette_selection(population, num_parents)
        else:
            raise ValueError(f"Unknown selection type: {self.selection_type}")
    
    def _tournament_selection(self, population: List[AllocationGenome], num_parents: int) -> List[AllocationGenome]:
        """锦标赛选择"""
        parents = []
        for _ in range(num_parents):
            # 随机选择tournament_size个个体
            tournament = random.sample(population, min(self.tournament_size, len(population)))
            # 选择适应度最高的
            winner = max(tournament, key=lambda x: x.get_average_fitness())
            parents.append(winner)
        return parents
    
    def _elite_selection(self, population: List[AllocationGenome], num_parents: int) -> List[AllocationGenome]:
        """精英选择"""
        # 按适应度排序
        sorted_pop = sorted(population, key=lambda x: x.get_average_fitness(), reverse=True)
        return sorted_pop[:num_parents]
    
    def _roulette_selection(self, population: List[AllocationGenome], num_parents: int) -> List[AllocationGenome]:
        """轮盘赌选择"""
        fitnesses = [max(0.001, x.get_average_fitness()) for x in population]  # 避免负值
        total_fitness = sum(fitnesses)
        
        if total_fitness == 0:
            return random.sample(population, min(num_parents, len(population)))
        
        probabilities = [f / total_fitness for f in fitnesses]
        parents = []
        
        for _ in range(num_parents):
            r = random.random()
            cumsum = 0
            for i, prob in enumerate(probabilities):
                cumsum += prob
                if r <= cumsum:
                    parents.append(population[i])
                    break
            else:
                parents.append(population[-1])  # 兜底选择最后一个
        
        return parents


class MutationOperator:
    """变异算子"""
    
    def __init__(self, mutation_rate: float = 0.1, mutation_strength: float = 0.1):
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
    
    def mutate(self, genome: AllocationGenome) -> AllocationGenome:
        """对基因进行变异"""
        mutated_genome = genome.clone()
        mutated_params = {}
        
        for name, param in genome.parameters.items():
            if random.random() < self.mutation_rate:
                # 高斯噪声变异
                noise = th.randn_like(param) * self.mutation_strength
                mutated_params[name] = param + noise
            else:
                mutated_params[name] = param.clone()
        
        mutated_genome.update_parameters(mutated_params)
        return mutated_genome
    
    def adaptive_mutate(self, genome: AllocationGenome, generation: int, max_generations: int) -> AllocationGenome:
        """自适应变异（随代数递减变异强度）"""
        # 计算自适应变异强度
        progress = generation / max_generations
        adaptive_strength = self.mutation_strength * (1 - progress * 0.5)  # 逐渐减小
        adaptive_rate = self.mutation_rate * (1 - progress * 0.3)  # 逐渐减小
        
        mutated_genome = genome.clone()
        mutated_params = {}
        
        for name, param in genome.parameters.items():
            if random.random() < adaptive_rate:
                # 高斯噪声变异
                noise = th.randn_like(param) * adaptive_strength
                mutated_params[name] = param + noise
            else:
                mutated_params[name] = param.clone()
        
        mutated_genome.update_parameters(mutated_params)
        return mutated_genome


class CrossoverOperator:
    """交叉算子"""
    
    def __init__(self, crossover_rate: float = 0.8, crossover_type: str = "uniform"):
        self.crossover_rate = crossover_rate
        self.crossover_type = crossover_type
    
    def crossover(self, parent1: AllocationGenome, parent2: AllocationGenome) -> Tuple[AllocationGenome, AllocationGenome]:
        """交叉操作"""
        if random.random() > self.crossover_rate:
            return parent1.clone(), parent2.clone()
        
        if self.crossover_type == "uniform":
            return self._uniform_crossover(parent1, parent2)
        elif self.crossover_type == "arithmetic":
            return self._arithmetic_crossover(parent1, parent2)
        else:
            raise ValueError(f"Unknown crossover type: {self.crossover_type}")
    
    def _uniform_crossover(self, parent1: AllocationGenome, parent2: AllocationGenome) -> Tuple[AllocationGenome, AllocationGenome]:
        """均匀交叉"""
        child1 = parent1.clone()
        child2 = parent2.clone()
        
        child1_params = {}
        child2_params = {}
        
        for name in parent1.parameters.keys():
            param1 = parent1.parameters[name]
            param2 = parent2.parameters[name]
            
            # 随机选择每个参数来自哪个父代
            mask = th.rand_like(param1) < 0.5
            child1_params[name] = th.where(mask, param1, param2)
            child2_params[name] = th.where(mask, param2, param1)
        
        child1.update_parameters(child1_params)
        child2.update_parameters(child2_params)
        
        return child1, child2
    
    def _arithmetic_crossover(self, parent1: AllocationGenome, parent2: AllocationGenome) -> Tuple[AllocationGenome, AllocationGenome]:
        """算术交叉"""
        child1 = parent1.clone()
        child2 = parent2.clone()
        
        alpha = random.uniform(0, 1)  # 交叉系数
        
        child1_params = {}
        child2_params = {}
        
        for name in parent1.parameters.keys():
            param1 = parent1.parameters[name]
            param2 = parent2.parameters[name]
            
            # 算术交叉：child = alpha * parent1 + (1-alpha) * parent2
            child1_params[name] = alpha * param1 + (1 - alpha) * param2
            child2_params[name] = (1 - alpha) * param1 + alpha * param2
        
        child1.update_parameters(child1_params)
        child2.update_parameters(child2_params)
        
        return child1, child2
