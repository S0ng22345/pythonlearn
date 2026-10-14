import gymnasium as gym     #导入强化学习标准环境库
from stable_baselines3 import PPO   #导入强化学习算法库POP算法

env = gym.make("CartPole-v1")       #搭建环境；不开启图形渲染窗口
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0001)    #创建POP模型实例，指定策略网络结构，
                                                                        # 传入环境，日志详细程度，学习率
model.learn(total_timesteps=50000)      #训练50000个时间步
model.save("ppo_cartpole")      #保存模型