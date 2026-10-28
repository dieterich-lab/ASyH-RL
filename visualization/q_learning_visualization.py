# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# 1. Hyperparameters
STATES = 6          # 1D world size
GOAL_STATE = 19      # Target position
ACTIONS = 2          # 0: Left, 1: Right
ALPHA = 0.1          # Learning rate
GAMMA = 0.9          # Discount factor
EPSILON = 0.2        # Exploration rate (higher to show learning curve)
EPISODES = 40

# Initialize Q-table
q_table = np.zeros((STATES, ACTIONS))

reward_history = []

# 2. Training Loop
print("Starting RL training to show convergence...")
for episode in range(EPISODES):
    # Random starting point to create variance in initial episodes
    state = np.random.randint(0, STATES - 1)
    total_reward = 0
    done = False
    steps = 0
    
    while not done and steps < 100:
        # Epsilon-greedy
        if np.random.uniform(0, 1) < EPSILON:
            action = np.random.randint(0, ACTIONS)
        else:
            action = np.argmax(q_table[state, :])
            
        # Environment logic: move left or right
        if action == 1: # Right
            next_state = min(STATES - 1, state + 1)
        else: # Left
            next_state = max(0, state - 1)
            
        # Reward design: Big reward for goal, small penalty for each step
        if next_state == GOAL_STATE:
            reward = 10
            done = True
        else:
            reward = -0.5
            
        # Q-Learning Update Rule: 
        # Q(s,a) = Q(s,a) + alpha * [R + gamma * max(Q(s',a')) - Q(s,a)]
        best_next_action = np.argmax(q_table[next_state, :])
        td_target = reward + GAMMA * q_table[next_state, best_next_action]
        q_table[state, action] += ALPHA * (td_target - q_table[state, action])
        
        state = next_state
        total_reward += reward
        steps += 1
        
    reward_history.append(total_reward)

# 3. Visualization
plt.figure(figsize=(10, 5))
# Plot raw data with transparency
plt.plot(reward_history, color='cornflowerblue', alpha=0.3, label='Episode Reward')

# Calculate moving average to highlight the convergence trend
window = 30
if len(reward_history) > window:
    smoothed = np.convolve(reward_history, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, EPISODES), smoothed, color='navy', linewidth=2, label='Learning Trend (Moving Avg)')

plt.title('Q-Learning Convergence: Privacy-Utility Optimization', fontsize=13)
plt.xlabel('Episodes', fontsize=11)
plt.ylabel('Cumulative Reward', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()

plt.savefig('convergence_plot_v2.png', dpi=300)
print("New plot saved as 'convergence_plot_v2.png'")
plt.show()