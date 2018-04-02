# Reinforcement_Learning_PPO

## Comparison of state-of-the-art reinforcement learning algorithms :
- Trust Region Policy Optimization (https://arxiv.org/pdf/1502.05477.pdf)
- Proximal Policy Optimization (https://arxiv.org/abs/1707.06347)
- Sample Efficient Actor-Critic with Experience Replay (https://arxiv.org/abs/1611.01224)
- Continuous control with deep reinforcement learning (https://arxiv.org/abs/1509.02971)
- Dueling Network Architectures for Deep Reinforcement Learning (https://arxiv.org/abs/1511.06581)

## Experimentations
Tests were done on 3 different environments: OpenAi-Gym (MountainCarContinuous), Mujoco (Reacher) and Atari (Breakout).
### OpenAi-Gym (MountainCarContinuous)
- DQN (`src/mountaincar-continuous/dqn` and `results/gym-mountaincarcontinuous/dqn`)
- DDPG (`src/mountaincar-continuous/ddpg` and `results/gym-mountaincarcontinuous/ddpg`)
- PPO (`src/mountaincar-continuous/ppo` and `results/gym-mountaincarcontinuous/ppo`)
### Mujoco (Reacher)
- DDPG (`src/baselines/baselines/ddpg` and `results/mujoco-reacher/ddpg`)
- TRPO (`src/baselines/baselines/trpo_mp` and `results/mujoco-reacher/trpo`)
- PPO (`src/baselines/baselines/ppo2` and `results/mujoco-reacher/ppo`)
### Atari (Breakout)
- ACER (`src/baselines/baselines/acer` and `results/atari-breakout/acer`)
- TRPO (`src/baselines/baselines/trpo_mp` and `results/atari-breakout/trpo`)
- PPO (`src/baselines/baselines/ppo2` and `results/atari-breakout/ppo`)

## Source code


## Requirements