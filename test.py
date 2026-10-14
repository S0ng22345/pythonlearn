import gymnasium as gym     #导入强化学习标准环境库
from stable_baselines3 import PPO       #导入强化学习算法库POP算法

env = gym.make("CartPole-v1", render_mode="human")      #搭建环境；渲染模式：人类可见
model = PPO.load("ppo_cartpole")        #加载之前训练的模型
obs, info = env.reset()        #初始化
for _ in range(500):        #最多尝试500步
    action, _ = model.predict(obs)      #AI根据当前状态决定动作
    obs, reward, terminated, truncated, info = env.step(action)     #执行动作
    if terminated or truncated:
        break                           #只看AI玩一局

env.close()     #关闭环境