# Reinforcement_Learning_PPO

## Comparison of state-of-the-art reinforcement learning algorithms :
- Trust Region Policy Optimization (https://arxiv.org/pdf/1502.05477.pdf)
- Proximal Policy Optimization (https://arxiv.org/abs/1707.06347)
- Sample Efficient Actor-Critic with Experience Replay (https://arxiv.org/abs/1611.01224)
- Continuous control with deep reinforcement learning (https://arxiv.org/abs/1509.02971)
- Dueling Network Architectures for Deep Reinforcement Learning (https://arxiv.org/abs/1511.06581)

## Other refence
- Deep Reinforcement Learning that Matters (https://arxiv.org/pdf/1709.06560.pdf)

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
To get the source code, execute the following commands:
```
git clone https://github.com/lajoiepy/Reinforcement_Learning_PPO.git
cd Reinforcement_Learning_PPO
git submodule init
git submodule update
```
- The source code in `src/baselines` is a fork of https://github.com/openai/baselines.
- The source code in `src/mountaincar-continuous/ddpg` is mostly from https://github.com/lirnli/OpenAI-gym-solutions/blob/master/Continuous_Deep_Deterministic_Policy_Gradient_Net/DDPG%20Class%20ver2.ipynb
- The source code in `src/mountaincar-continuous/dqn` is inspired from http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html 
- The source code in `src/mountaincar-continuous/ppo` is mostly from https://github.com/tpbarron/pytorch-ppo

## Requirements
- Pytorch for `src/mountaincar-continuous/dqn` and `src/mountaincar-continuous/ppo`.
- Tensorflow for `src/mountaincar-continuous/ddpg` and `src/baselines`.
- Gym for `src/mountaincar-continuous`.
- Mujoco and Atari for `src/baselines`.