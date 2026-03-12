from .agent import Agent
from .allocation_critics import *
from .allocation_policies import *
from functools import partial

ALLOC_CRITIC_REGISTRY = {}
ALLOC_CRITIC_REGISTRY['standard'] = StandardAllocCritic
ALLOC_CRITIC_REGISTRY['a2c'] = A2CAllocCritic
ALLOC_CRITIC_REGISTRY['ppo'] = A2CAllocCritic  # PPO使用与A2C相同的Critic

ALLOC_POLICY_REGISTRY = {}
ALLOC_POLICY_REGISTRY['autoreg'] = AutoregressiveAllocPolicy
ALLOC_POLICY_REGISTRY['a2c'] = A2CAllocPolicy
ALLOC_POLICY_REGISTRY['ppo'] = A2CAllocPolicy  # PPO使用与A2C相同的Policy
