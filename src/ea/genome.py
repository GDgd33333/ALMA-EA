import torch as th  
import torch.nn as nn  
import copy  
from typing import Dict, Any, List  
import numpy as np  


class AllocationGenome:
    """
    分配器基因编码类，封装分配器网络参数作为进化个体
    
    该类将分配器网络（AutoregressiveAllocPolicy）封装为进化算法中的个体，
    提供参数管理、适应度跟踪、克隆等功能，支持进化操作。
    """
    
    def __init__(self, alloc_policy: nn.Module, genome_id: int = 0):
        """
        初始化分配器基因
        
        Args:
            alloc_policy: 分配器网络（AutoregressiveAllocPolicy），用作基因组的模板
            genome_id: 基因ID，用于唯一标识该基因组
        """
        self.genome_id = genome_id  # 基因组唯一标识符
        self.alloc_policy = copy.deepcopy(alloc_policy)  # 深拷贝分配器网络，避免共享参数
        self.fitness = 0.0  # 当前适应度值，初始为0
        self.fitness_history = []  # 适应度历史记录列表
        self.evaluation_count = 0  # 评估次数计数器
        
        # 提取网络参数用于进化操作
        self.parameters = {}  # 参数字典，存储网络参数的副本
        self._extract_parameters()  # 提取网络参数到参数字典中
    
    def _extract_parameters(self):
        """
        提取网络参数到字典中
        
        遍历分配器网络的所有参数，将参数数据克隆到参数字典中，
        用于进化操作时的参数访问和修改。
        """
        for name, param in self.alloc_policy.named_parameters():  # 遍历网络的所有命名参数
            self.parameters[name] = param.data.clone()  # 克隆参数数据到参数字典
    
    def update_parameters(self, new_params: Dict[str, th.Tensor]):
        """
        更新网络参数
        
        Args:
            new_params: 新的参数字典，包含要更新的参数
        """
        for name, param in new_params.items():  # 遍历新参数字典
            if name in self.parameters:  # 如果参数名存在于当前参数字典中
                self.parameters[name].copy_(param)  # 更新参数字典中的参数
                # 同步到实际网络
                for n, p in self.alloc_policy.named_parameters():  # 遍历网络参数
                    if n == name:  # 找到对应的网络参数
                        p.data.copy_(param)  # 更新网络参数数据
                        break  # 找到后跳出循环
    
    def get_parameters(self) -> Dict[str, th.Tensor]:
        """
        获取网络参数
        
        Returns:
            Dict[str, th.Tensor]: 参数字典的副本，避免外部修改影响原始参数
        """
        return {name: param.clone() for name, param in self.parameters.items()}  # 返回参数字典的深拷贝
    
    def clone(self) -> 'AllocationGenome':
        """
        克隆基因
        
        创建一个新的基因组实例，复制当前基因组的所有属性，
        用于进化算法中的个体复制操作。
        
        Returns:
            AllocationGenome: 新的基因组实例
        """
        new_genome = AllocationGenome(self.alloc_policy, self.genome_id)  # 创建新的基因组实例
        new_genome.fitness = self.fitness  # 复制适应度值
        new_genome.fitness_history = self.fitness_history.copy()  # 复制适应度历史（浅拷贝列表）
        new_genome.evaluation_count = self.evaluation_count  # 复制评估次数
        return new_genome
    
    def reset_fitness(self):
        """
        重置适应度
        
        将适应度值和评估次数重置为初始状态，
        用于新一轮评估开始前的清理工作。
        """
        self.fitness = 0.0  # 重置适应度值为0
        self.evaluation_count = 0  # 重置评估次数为0
    
    def add_fitness(self, reward: float):
        """
        添加适应度值
        
        Args:
            reward: 单次评估的奖励值
        """
        self.fitness += reward  # 累加奖励到总适应度
        self.evaluation_count += 1  # 增加评估次数计数
    
    def get_average_fitness(self) -> float:
        """
        获取平均适应度
        
        Returns:
            float: 平均适应度值，如果未进行评估则返回0.0
        """
        if self.evaluation_count == 0:  # 如果未进行评估
            return 0.0  # 返回0.0
        return self.fitness / self.evaluation_count  # 返回总适应度除以评估次数
    
    def update_fitness_history(self):
        """
        更新适应度历史
        
        将当前平均适应度添加到历史记录中，
        并保持历史记录在合理范围内（最多100条记录）。
        """
        avg_fitness = self.get_average_fitness()  # 获取当前平均适应度
        self.fitness_history.append(avg_fitness)  # 添加到历史记录
        # 保持历史记录在合理范围内
        if len(self.fitness_history) > 100:  # 如果历史记录超过100条
            self.fitness_history = self.fitness_history[-100:]  # 只保留最近100条记录
    
    def get_parameter_vector(self) -> th.Tensor:
        """
        获取参数向量（用于某些进化操作）
        
        将所有网络参数展平并连接成一个一维向量，
        用于某些需要向量化操作的进化算法。
        
        Returns:
            th.Tensor: 一维参数向量
        """
        param_list = []  # 参数列表，用于存储展平后的参数
        for param in self.parameters.values():  # 遍历所有参数
            param_list.append(param.flatten())  # 将参数展平并添加到列表
        return th.cat(param_list)  # 连接所有展平的参数为一个向量
    
    def set_parameter_vector(self, param_vector: th.Tensor):
        """
        从参数向量设置参数
        
        将一维参数向量重新分割并设置到对应的网络参数中，
        用于从向量化表示恢复网络参数。
        
        Args:
            param_vector: 一维参数向量
        """
        start_idx = 0  # 起始索引
        for name, param in self.parameters.items():  # 遍历参数字典
            param_size = param.numel()  # 获取参数的元素数量
            # 从向量中提取对应参数的数据并重塑为原始形状
            param_data = param_vector[start_idx:start_idx + param_size].reshape(param.shape)
            param.copy_(param_data)  # 更新参数字典中的参数
            start_idx += param_size  # 更新起始索引
            
            # 同步到实际网络
            for n, p in self.alloc_policy.named_parameters():  # 遍历网络参数
                if n == name:  # 找到对应的网络参数
                    p.data.copy_(param_data)  # 更新网络参数数据
                    break  # 找到后跳出循环
    
    def __str__(self):
        """
        字符串表示方法
        
        Returns:
            str: 基因组的字符串表示，包含ID、适应度和评估次数
        """
        return f"AllocationGenome(id={self.genome_id}, fitness={self.fitness:.4f}, evals={self.evaluation_count})"
