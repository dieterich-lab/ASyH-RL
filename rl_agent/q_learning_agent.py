"""
Q-Learning 代理
用于学习最优的评估策略
"""

import numpy as np
import pickle
from typing import Tuple, List, Dict


class QLearningAgent:
    """Q-Learning 强化学习代理"""
    
    def __init__(self, state_size: int, action_size: int, 
                 learning_rate: float = 0.1, 
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                #  epsilon_decay: float = 0.995,
                #  epsilon_min: float = 0.01
                ):
        """
        初始化Q-Learning代理
        
        Args:
            state_size: 状态空间大小
            action_size: 动作空间大小
            learning_rate: 学习率
            discount_factor: 折扣因子
            epsilon: 探索率
            epsilon_decay: 探索率衰减
            epsilon_min: 最小探索率
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        # self.epsilon_decay = epsilon_decay
        # self.epsilon_min = epsilon_min
        
        self.q_table = {}
        
        # 训练历史
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'epsilon': []
        }
    
    def _discretize_state(self, state: np.ndarray, bins: int = 10) -> Tuple:
        """
        将连续状态离散化
        
        Args:
            state: 连续状态
            bins: 离散化的箱数
            
        Returns:
            离散化后的状态元组
        """
        discretized = []
        for value in state:
            # 将值映射到[0, bins-1]的整数
            discretized_value = int(min(bins - 1, max(0, value * bins)))
            discretized.append(discretized_value)
        return tuple(discretized)
    
    def get_q_value(self, state: Tuple, action: int) -> float:
        """
        获取Q值
        
        Args:
            state: 状态
            action: 动作
            
        Returns:
            Q值
        """
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0.0
        return self.q_table[(state, action)]
    
    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        选择动作（ε-greedy策略）
        
        Args:
            state: 当前状态
            training: 是否为训练模式
            
        Returns:
            选择的动作
        """
        state_discrete = self._discretize_state(state)
        
        # ε-greedy策略
        if training and np.random.random() < self.epsilon:
            # 探索：随机选择动作
            return np.random.randint(0, self.action_size)
        else:
            # 利用：选择Q值最大的动作
            q_values = [self.get_q_value(state_discrete, a) for a in range(self.action_size)]
            return int(np.argmax(q_values))
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool):
        """
        更新Q表
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        state_discrete = self._discretize_state(state)
        next_state_discrete = self._discretize_state(next_state)
        
        # 当前Q值
        current_q = self.get_q_value(state_discrete, action)
        
        # 计算目标Q值
        if done:
            target_q = reward
        else:
            # 下一状态的最大Q值
            next_q_values = [self.get_q_value(next_state_discrete, a) 
                           for a in range(self.action_size)]
            max_next_q = max(next_q_values)
            target_q = reward + self.discount_factor * max_next_q
        
        # 更新Q值
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_table[(state_discrete, action)] = new_q
    
    # def decay_epsilon(self):
    #     """衰减探索率"""
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay
    
    def save(self, filepath: str):
        """保存模型"""
        model_data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'training_history': self.training_history,
            'state_size': self.state_size,
            'action_size': self.action_size
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath: str):
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_table = model_data['q_table']
        self.epsilon = model_data['epsilon']
        self.training_history = model_data['training_history']
        self.state_size = model_data['state_size']
        self.action_size = model_data['action_size']

