import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("CartPole-v1", render_mode="human")
model = PPO.load("ppo_cartpole")
obs, info = env.reset()
for _ in range(500):
    action, _ = model.predict(obs) # AI根据当前状态决定动作
    obs, reward, terminated, truncated, info = env.step(action) # 执行动作
    if terminated or truncated:
        break

env.close()