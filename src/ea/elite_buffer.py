import torch as th
import numpy as np
from typing import Dict, Any, Optional, List
from collections import deque
import copy


class EliteBuffer:
    """
    精英缓存池：存储EA评估阶段发现的成功/高质量任务分配序列
    
    作用：
    - 存储EA找到的好分配结构
    - 供PPO通过蒸馏损失学习这些好结构
    - 不参与PPO的on-policy计算
    """
    
    def __init__(self,
                 max_size: int = 200,  # 最大容量（100-200条足够）
                 min_score: float = 0.0,  # 最小分数阈值（用于筛选精英）
                 device: str = "cuda",
                 use_relative_ranking: bool = True,  # 是否使用相对排名策略（处理低成功率情况）
                 top_k_percent: float = 0.3):  # 保存前K%的episode（用于相对排名）
        """
        初始化精英缓存池
        
        Args:
            max_size: 最大容量
            min_score: 最小分数阈值（episode return或success标志）
            device: 计算设备
            use_relative_ranking: 是否使用相对排名策略（如果成功率低，保存前K%的episode）
            top_k_percent: 保存前K%的episode（0.3表示前30%）
        """
        self.max_size = max_size
        self.min_score = min_score
        self.device = device
        self.use_relative_ranking = use_relative_ranking
        self.top_k_percent = top_k_percent
        
        # 使用deque实现FIFO（先进先出）
        self.buffer: deque = deque(maxlen=max_size)
        
        # 相对排名相关：记录最近N个episode的reward
        self.recent_rewards: deque = deque(maxlen=100)  # 最多记录100个episode的reward
        
        # 统计信息
        self.stats = {
            'total_added': 0,
            'total_rejected': 0,
            'current_size': 0,
            'added_by_success': 0,  # 因成功而添加的数量
            'added_by_threshold': 0,  # 因reward阈值而添加的数量
            'added_by_ranking': 0,  # 因相对排名而添加的数量
        }
        
        # 修复坑A：明确哪些key有batch维度
        # 这些key的第一维是batch维度，需要取[0]来获取单样本
        self.batched_keys = set([
            "entities", "entity_mask", "entity2task_mask", "task_mask", "obs_mask",
            "last_alloc", "task_nonag_counts", "task_ag_counts",
            # 根据实际meta_inputs中的key添加
        ])
    
    def add(self,
            meta_inputs: Dict[str, th.Tensor],
            alloc_order: th.Tensor,
            alloc_actions: th.Tensor,
            score: float,
            episode_reward: Optional[float] = None):
        """
        添加一条精英样本
        
        Args:
            meta_inputs: 上层分配网络forward所需输入
                - 必须包含：entities, entity_mask, entity2task_mask, task_mask, obs_mask, last_alloc等
            alloc_order: (na,) 或 (bs, na) - 自回归分配顺序
            alloc_actions: (na, nt) 或 (bs, na, nt) - 分配动作序列（one-hot编码）
            score: 成功标志(1.0)或episode return（用于筛选）
            episode_reward: episode的reward（用于相对排名，可选）
        """
        # 记录reward（用于相对排名）
        if episode_reward is not None:
            self.recent_rewards.append(episode_reward)
        
        # 判断是否保存（混合策略）
        should_save = False
        save_reason = None
        
        # 策略1：绝对成功（score=1.0表示battle_won=True）
        if score >= 1.0:
            should_save = True
            save_reason = 'success'
        # 策略2：reward阈值
        elif score >= self.min_score:
            should_save = True
            save_reason = 'threshold'
        # 策略3：相对排名（如果启用且最近有足够的数据）
        elif self.use_relative_ranking and len(self.recent_rewards) >= 20 and episode_reward is not None:
            sorted_rewards = sorted(self.recent_rewards, reverse=True)
            rank_index = max(0, int(len(sorted_rewards) * self.top_k_percent) - 1)
            if rank_index < len(sorted_rewards) and episode_reward >= sorted_rewards[rank_index]:
                should_save = True
                save_reason = 'ranking'
        
        if not should_save:
            self.stats['total_rejected'] += 1
            return
        
        # 处理batch维度（确保是单样本）
        if alloc_order.dim() == 2:
            alloc_order = alloc_order[0]  # (bs, na) -> (na,)
        if alloc_actions.dim() == 3:
            alloc_actions = alloc_actions[0]  # (bs, na, nt) -> (na, nt)
        
        # 确保tensor在正确的设备上并detach
        alloc_order = alloc_order.to(self.device).detach().clone()
        alloc_actions = alloc_actions.to(self.device).detach().clone()
        
        # 深拷贝meta_inputs（确保不共享内存）
        elite_sample = {
            'meta_inputs': {},
            'alloc_order': alloc_order,
            'alloc_actions': alloc_actions,
            'score': score,
        }
        
        # 复制meta_inputs中的所有tensor
        # 修复坑A：只在明确有batch维的key上取[0]
        for key, value in meta_inputs.items():
            if isinstance(value, th.Tensor):
                v = value
                # 只在明确标记为有batch维的key上，且确实有batch维时，才取[0]
                if key in self.batched_keys and v.dim() >= 1 and v.shape[0] > 1:
                    v = v[0]  # 取第一个样本
                elite_sample['meta_inputs'][key] = v.to(self.device).detach().clone()
            else:
                elite_sample['meta_inputs'][key] = copy.deepcopy(value)
        
        # 添加到buffer
        self.buffer.append(elite_sample)
        self.stats['total_added'] += 1
        self.stats['current_size'] = len(self.buffer)
        
        # 更新统计信息
        # 格式化reward字符串（避免格式化字符串错误）
        reward_str = f"{episode_reward:.2f}" if episode_reward is not None else "N/A"
        
        if save_reason == 'success':
            self.stats['added_by_success'] += 1
            # 打印成功保存的信息
            print(f"[EliteBuffer] ✅ 成功策略已保存到精英缓存池！"
                  f" score={score:.2f}, reward={reward_str}, "
                  f"当前缓存池大小={len(self.buffer)}/{self.max_size}")
        elif save_reason == 'threshold':
            self.stats['added_by_threshold'] += 1
            print(f"[EliteBuffer] 📊 高质量策略已保存（阈值）！"
                  f" score={score:.2f}, reward={reward_str}, "
                  f"当前缓存池大小={len(self.buffer)}/{self.max_size}")
        elif save_reason == 'ranking':
            self.stats['added_by_ranking'] += 1
            print(f"[EliteBuffer] 🏆 相对排名策略已保存！"
                  f" score={score:.2f}, reward={reward_str}, "
                  f"当前缓存池大小={len(self.buffer)}/{self.max_size}")
    
    def sample(self, batch_size: int = 16) -> Optional[Dict[str, Any]]:
        """
        从buffer中采样一批精英样本
        
        关键修复：确保所有字段的batch数一致，避免错位
        - 先确定一个"有效样本索引集合valid_idx"
        - 然后对所有key都用同一批样本
        
        Args:
            batch_size: 批次大小（16-32足够）
        
        Returns:
            如果buffer为空返回None，否则返回批次数据
        """
        if len(self.buffer) == 0:
            return None
        
        # 随机采样
        sample_size = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), sample_size, replace=False)
        samples = [self.buffer[i] for i in indices]
        
        if len(samples) == 0:
            return None
        
        # 1) 先用第一条样本确定target shapes（最简单）
        ref = samples[0]
        target_ao_shape = ref["alloc_order"].shape        # (na,)
        target_aa_shape = ref["alloc_actions"].shape      # (na, nt)
        
        # 处理可能的伪batch维度
        if len(target_ao_shape) == 2 and target_ao_shape[0] == 1:
            target_ao_shape = target_ao_shape[1:]
        if len(target_aa_shape) == 3 and target_aa_shape[0] == 1:
            target_aa_shape = target_aa_shape[1:]
        
        meta_keys = list(ref["meta_inputs"].keys())
        target_meta_shapes = {}
        for k in meta_keys:
            v = ref["meta_inputs"][k]
            if isinstance(v, th.Tensor):
                # 统一掉可能的(1,...)伪batch
                if v.dim() > 1 and v.shape[0] == 1:
                    v = v.squeeze(0)
                target_meta_shapes[k] = v.shape
        
        # 2) 选出所有字段都一致的样本索引
        valid = []
        for s in samples:
            ao = s["alloc_order"]
            aa = s["alloc_actions"]
            
            # 处理可能的batch维度
            if ao.dim() == 2:
                ao = ao[0]
            if aa.dim() == 3:
                aa = aa[0]
            
            # 检查alloc_order和alloc_actions形状
            if ao.shape != target_ao_shape:
                continue
            if aa.shape != target_aa_shape:
                continue
            
            # 检查所有meta_inputs的key形状
            ok = True
            for k in meta_keys:
                v = s["meta_inputs"][k]
                if isinstance(v, th.Tensor):
                    if v.dim() > 1 and v.shape[0] == 1:
                        v = v.squeeze(0)
                    if k in target_meta_shapes and v.shape != target_meta_shapes[k]:
                        ok = False
                        break
            if ok:
                valid.append(s)
        
        if len(valid) == 0:
            if len(samples) > 0:
                print(f"Warning: EliteBuffer.sample() - no valid samples found. "
                      f"Expected ao_shape={target_ao_shape}, aa_shape={target_aa_shape}. "
                      f"Returning None.")
            return None
        
        if len(valid) < len(samples):
            print(f"Warning: EliteBuffer.sample() - using {len(valid)}/{len(samples)} valid samples "
                  f"(skipped {len(samples) - len(valid)} inconsistent samples).")
        
        # 3) stack：所有字段用同一批valid样本
        batch = {"meta_inputs": {}, "alloc_order": None, "alloc_actions": None, "score": []}
        
        for k in meta_keys:
            vals = [s["meta_inputs"][k] for s in valid]
            if all(isinstance(v, th.Tensor) for v in vals):
                norm = []
                for v in vals:
                    if v.dim() > 1 and v.shape[0] == 1:
                        v = v.squeeze(0)
                    norm.append(v)
                batch["meta_inputs"][k] = th.stack(norm, dim=0)
            else:
                batch["meta_inputs"][k] = vals
        
        aos = []
        aas = []
        for s in valid:
            ao = s["alloc_order"]
            aa = s["alloc_actions"]
            if ao.dim() == 2:
                ao = ao[0]
            if aa.dim() == 3:
                aa = aa[0]
            aos.append(ao)
            aas.append(aa)
            batch["score"].append(s["score"])
        
        batch["alloc_order"] = th.stack(aos, dim=0)        # (bs, na)
        batch["alloc_actions"] = th.stack(aas, dim=0)      # (bs, na, nt)
        
        return batch
    
    def clear(self):
        """清空buffer"""
        self.buffer.clear()
        self.stats['current_size'] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
    
    def __len__(self):
        """返回buffer大小"""
        return len(self.buffer)
    
    def __repr__(self):
        return (f"EliteBuffer(size={len(self.buffer)}/{self.max_size}, "
                f"added={self.stats['total_added']}, rejected={self.stats['total_rejected']}, "
                f"by_success={self.stats['added_by_success']}, "
                f"by_threshold={self.stats['added_by_threshold']}, "
                f"by_ranking={self.stats['added_by_ranking']})")


