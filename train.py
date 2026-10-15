import gymnasium as gym     #导入强化学习标准环境库
from stable_baselines3 import PPO   #导入强化学习算法库POP算法

env = gym.make("CartPole-v1")       #搭建环境；不开启图形渲染窗口
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_cartpole_tensorboard/")
model.learn(total_timesteps=100000) # 改成10万步
model.save("ppo_cartpole_100k")