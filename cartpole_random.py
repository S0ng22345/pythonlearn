import gymnasium as gym     #导入强化学习标准环境库

env = gym.make("CartPole-v1",render_mode='human')       #搭建环境；渲染模式：人类可见
obs, info = env.reset()     #初始化
for _ in range(500):        #训练500个时间步
    action = env.action_space.sample()      #随机取样
    obs, reward, terminated, truncated, info = env.step(action)     #action提交給环境，环境返回下一步状态
    if terminated or truncated:     #判断是否结束（前者为游戏失败）（后者为游戏结束）
        obs, info = env.reset()     #初始化
env.close()     #关闭环境