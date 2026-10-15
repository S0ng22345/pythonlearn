import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")
model = PPO.load("ppo_cartpole_100k")

# 跑 10 局，统计得分
episode_rewards = []
for episode in range(10):
    obs, info = env.reset()
    total_steps = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_steps += 1
        done = terminated or truncated

    episode_rewards.append(total_steps)
    print(f"第 {episode + 1} 局，坚持了 {total_steps} 步")

env.close()

# 计算平均分
average_steps = np.mean(episode_rewards)
print(f"\n10局平均坚持步数: {average_steps:.2f} 步")