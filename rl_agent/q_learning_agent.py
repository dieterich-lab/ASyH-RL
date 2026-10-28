import numpy as np
import pickle
from typing import Tuple, List, Dict


class QLearningAgent:
    """Q-Learning agent module"""
    
    def __init__(self, state_size: int, action_size: int, 
                 learning_rate: float = 0.1, 
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                #  epsilon_decay: float = 0.995,
                #  epsilon_min: float = 0.01
                ):
        """
        The constructor of the Q-Learning agent
        
        Args:
            state_size: minimal number of var-s needed to describe a current state
            action_size: how many different action options exist at the moment
            learning_rate: it decided how much will the agent update its q-table after each iteration
            discount_factor: the discount on future rewards
            epsilon: the parameter which dictates if the agent will favor explorative or exploitative policy
            epsilon_decay: how much will epsilon reduce after each iteration of the agent?
            epsilon_min: the minimal possible epsilon
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        # self.epsilon_decay = epsilon_decay
        # self.epsilon_min = epsilon_min
        
        self.q_table = {}
        
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'epsilon': []
        }
    
    def _discretize_state(self, state: np.ndarray, bins: int = 10) -> Tuple:
        """
        Break continuous variables that characterize a state into 
        discrete ones
        
        Args:
            state: np.ndarray, the state of agent's environment at the moment,
            bins: int, histogram bins
            
        Returns:
            A tuple of discretize states
        """
        discretized = []
        for value in state:
            discretized_value = int(min(bins - 1, max(0, value * bins)))
            discretized.append(discretized_value)
        return tuple(discretized)
    
    def get_q_value(self, state: Tuple, action: int) -> float:
        """
        Pull a Q-value from the Q-table based on taken action and curr. state
        
        Args:
            state: Tuple
            action: int
            
        Returns:
            Q: float
        """
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0.0
        return self.q_table[(state, action)]
    
    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        The function decides on a next action （following ε-greedy strategy）
        
        Args:
            state: np.ndarray, the current state of an environment,
            training: bool,  if an agent is in training mode
            
        Returns:
            action: integer
        """
        state_discrete = self._discretize_state(state)
        
        # ε-greedy strategy
        if training and np.random.random() < self.epsilon:
            # take an explorative step
            return np.random.randint(0, self.action_size)
        else:
            # exploitative step,  choose a best action using the q-table
            q_values = [self.get_q_value(state_discrete, a) for a in range(self.action_size)]
            return int(np.argmax(q_values))
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool):
        """
        
        Args:
            state: np.ndarray,
            action: int,  an index of a taken action (among available actions)
            reward: float, 
            next_state: np.ndarray,
            done: bool,  if the feedback is complete
        """
        state_discrete = self._discretize_state(state)
        next_state_discrete = self._discretize_state(next_state)
        
        # update the Q-value
        current_q = self.get_q_value(state_discrete, action)
        
        if done:
            target_q = reward
        else:
            next_q_values = [self.get_q_value(next_state_discrete, a) 
                           for a in range(self.action_size)]
            max_next_q = max(next_q_values)
            target_q = reward + self.discount_factor * max_next_q
        
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_table[(state_discrete, action)] = new_q

    def save(self, filepath: str):
        """Save the model data and dump on a disk as pickle file"""
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
        """Load an agent model data from a saved pickle"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_table = model_data['q_table']
        self.epsilon = model_data['epsilon']
        self.training_history = model_data['training_history']
        self.state_size = model_data['state_size']
        self.action_size = model_data['action_size']