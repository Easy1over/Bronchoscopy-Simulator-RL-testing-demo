from stable_baselines3 import PPO
from MyNetwork import policy_kwargs_Mlt  # 引入外部定义的网络结构
from Env import BronchoscopyEnv  # 你的环境
import time

env = BronchoscopyEnv()

i=1

model = PPO(
    "MultiInputPolicy",
    env,
    policy_kwargs=policy_kwargs_Mlt,
    verbose=1,
    tensorboard_log=f"./logs/model{i}",
    learning_rate=0.0003,
    n_steps=128,
    batch_size=64,
    n_epochs=4,
    gamma=0.98,
    normalize_advantage=True,
    gae_lambda=0.90,
    ent_coef=0.2,
    vf_coef=0.5,
    max_grad_norm=0.5,
    clip_range=0.2,
    clip_range_vf=0.2,
    device="cuda",
    stats_window_size=100,
)
print(model.policy)

start_time = time.time()
try:
    # 从0开始训练模型
    model.learn(total_timesteps=500000)
except KeyboardInterrupt:
    # 当用户按下 Ctrl+C 中断训练时手动保存模型
    print("训练被中断，正在保存当前模型...")
    model.save(f"./model/ppo_model{i}")   #模型保存的路径
    print("模型已保存...")
finally:
    end_time = time.time()
    print(f"总训练时间: {end_time - start_time} 秒")

print("训练完成!")
model.save(f"./model/ppo_model{i}")
print("模型已保存!")


