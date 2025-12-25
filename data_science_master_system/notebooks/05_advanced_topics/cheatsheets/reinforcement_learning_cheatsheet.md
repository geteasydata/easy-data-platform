# 📋 Reinforcement Learning Cheatsheet

## 📌 Key Concepts
- **Agent**: Learner/decision maker
- **Environment**: World agent interacts with
- **State (s)**: Current situation
- **Action (a)**: What agent can do
- **Reward (r)**: Feedback signal
- **Policy (π)**: Strategy for choosing actions
- **Value (V/Q)**: Expected future reward

## 🛠️ Essential Code

### Q-Learning (Tabular)
```python
Q = np.zeros((n_states, n_actions))
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor

# Update rule
Q[state, action] += alpha * (
    reward + gamma * np.max(Q[next_state]) - Q[state, action]
)
```

### DQN (Deep Q-Network)
```python
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)
```

### Stable-Baselines3
```python
from stable_baselines3 import PPO

model = PPO('MlpPolicy', 'CartPole-v1', verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
action, _ = model.predict(obs)
```

## 📊 Algorithm Comparison
| Algorithm | Type | Actions | Use Case |
|-----------|------|---------|----------|
| Q-Learning | Value | Discrete | Simple |
| DQN | Value | Discrete | Atari games |
| A2C/A3C | Actor-Critic | Both | Parallel |
| PPO | Policy | Both | General |
| SAC | Policy | Continuous | Robotics |

## 📐 Key Formulas
```
Bellman Equation:
V(s) = max_a [R(s,a) + γ * V(s')]

Policy Gradient:
∇J(θ) = E[∇log π(a|s) * Q(s,a)]
```

## ⚠️ Common Pitfalls
| Problem | Solution |
|---------|----------|
| Unstable training | Use target network, smaller LR |
| Sparse rewards | Reward shaping |
| Sample inefficiency | Experience replay, prioritized |
| Exploration | ε-greedy, entropy bonus |

## 🚀 Production Tips
- Curriculum learning: Easy → Hard
- Sim2Real transfer
- Offline RL for safety-critical
- Multi-agent for complex systems
