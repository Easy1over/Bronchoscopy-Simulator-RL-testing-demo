import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import pickle
import time
#import Image_Processing as IP
#import torch
from filelock import FileLock
from gymnasium.spaces import Dict, Box

#D:\Code\RL_BiSST\RL_Bisst\data_exchange


# 编写强化学习环境的类
class BronchoscopyEnv(gym.Env):
    
    #初始化
    # 这个函数在创建环境时调用，用于初始化环境的参数和状态
    def __init__(self,render_mode=None):
        
        print("init now!!!!!!") #开始初始化
        
        # 1. 环境初始化
        # 在这里定义你的环境的基本参数和状态
        # 主要是状态空间 动作空间 回合限长
        # 继承gym的标准父类 
        super(BronchoscopyEnv, self).__init__()

        # 2. 定义动作空间
        # 动作空间是一个连续的空间，范围在[-1, 1]之间，表示旋转、弯曲的动作

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([ 1.0,  1.0,  1.0], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )


        self.observation_space = Dict({
    "image": Box(low=0, high=1, shape=(3, 224, 224), dtype=np.float32),
    "vector": Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
})

        # 4. 初始化其他环境变量 这部分可能需要补充
        self.max_steps = 500  # 最大步数（每回合） 这个是用于一轮不能探索太多的
        self.current_step = 0 # 当前步数（初始化）
        self.done = False  # 是否结束标志（对于每个回合 初始时置为false）
        self.delta_max = 0.1  # 设定最大增量（根据任务调整） 参数delta_max用于计算惩罚与限制弯曲变化范围（未启用）
        self.oldfinshirate=0 # 记录上一次的完成率（启用）
        print("init over") #初始化结束

    ##D:\Code\RL_BiSST\RL_Bisst\data_exchange
    # 重置环境
    # 这个函数在每个回合开始时调用，用于重置环境状态
    def reset(self,seed=None, options=None):
       
        print("reset now") #开始重置

        self.done = False #结束标志置为false
        over_flag=1 #结束标志 置为1 用于告诉仿真平台需要重置了
        self.current_step = 0  # 重置当前步数
        #self.last_theta = 0.0  # 初始旋转角度
        #self.last_phi = 0.0  # 初始弯曲角度
        # 将动作写入文件
        Ori_Myaction = {"action":[1,0,0],"over_flag":over_flag} #注意写进去的action是手柄！！！ y rx ry
        lock = FileLock("D:\Code\RL_BiSST\RL_Bisst\data_exchange\Myaction.pkl.lock")
        #print(1)
        with lock:
            with open('D:\Code\RL_BiSST\RL_Bisst\data_exchange\Myaction.pkl', 'wb') as f:
                pickle.dump(Ori_Myaction, f)

        #print(2)

        #读取观察空间文件里的内容
        while True:
            try:
                with open('D:\Code\RL_BiSST\RL_Bisst\data_exchange\Myobservation.pkl','rb') as f:
                    Myobservation = pickle.load(f) 
                if Myobservation['read_flag'] is not None:
                    observation_dict = Myobservation      #后续用observation_dict来代替
                    Myobservation['read_flag']= None      #清空标记文件 避免一次文件被多次使用
                    with open('D:\Code\RL_BiSST\RL_Bisst\data_exchange\Myobservation.pkl', 'wb') as f:
                        pickle.dump(Myobservation, f)
                    break  #读取到东西就跳出循环
            except Exception as e: #不断读取文件，直到读取到数据
                time.sleep(0.001)
                print("wait...")
                pass
        time.sleep(0.001)  #避免更新过快 
        #读取到了数据那么就进行后续操作，首先解耦数据
        image=observation_dict['image'] #图像数据 224x224x3
        Obv_11=observation_dict['Sensor11']  #传感器数据 11维度
        observation={
            "image": image,   # (3, 224, 224), float32
            "vector": self.obv_normalize(Obv_11)     # (11,), float32
        }
        self.oldfinshirate=0
        print("reset over")

        return observation,{} #继承gym.env的标准输出
    

    #  执行一步
    # 这个函数在每一步执行时调用，用于更新环境状态和计算奖励
    def step(self, action): #输入是网络预测的动作 这个动作的维度和大小范围会跟初始化里确定的一致 

        print("step now")
        #由于强化学习仿真平台是由手柄控制平台改写过来的，考虑到一致性 这里将动作空间的输入映射手柄输入，后续操作等效
        over_flag=0 #技术标志 置为0 step说明还在继续进行
        action[1],action[2]=self._project_to_unit_circle(action[1], action[2])  # 将弯曲和进退投影到单位圆上

        Myaction = {"action":action,"over_flag":over_flag} #将动作写入 等待后续相应
        lock = FileLock("D:\Code\RL_BiSST\RL_Bisst\data_exchange\Myaction.pkl.lock")
        with lock:
            with open('D:\Code\RL_BiSST\RL_Bisst\data_exchange\Myaction.pkl', 'wb') as f:
                pickle.dump(Myaction, f)

        #读取观察空间文件里的内容
        while True:
            try:
                with open('D:\Code\RL_BiSST\RL_Bisst\data_exchange\Myobservation.pkl','rb') as f:
                    Myobservation = pickle.load(f) 
                if Myobservation['read_flag'] is not None:
                    observation_dict = Myobservation             #后续用observation来代替
                    Myobservation['read_flag']= None             #清空文件 避免一次文件被多次使用
                    with open('D:\Code\RL_BiSST\RL_Bisst\data_exchange\Myobservation.pkl', 'wb') as f:
                        pickle.dump(Myobservation, f)
                    break  #读取到东西就跳出循环
            except Exception as e:
                time.sleep(0.001)
                pass

        self.current_step += 1  # 更新步数
        time.sleep(0.001)  #避免更新过快 

        #读取到了数据那么就进行后续操作，首先解耦数据
        image=observation_dict['image']
        Obv_11=observation_dict['Sensor11']  #计算距离终点向量差：观察空间
        Handdistance=observation_dict['deviation_position'] #计算奖励 距离单条中心线距离
        update_flag=observation_dict['update_flag'] #计算奖励，判断游戏状态

        current_positiontotarget=[Obv_11[0]-Obv_11[3],Obv_11[1]-Obv_11[4],Obv_11[2]-Obv_11[5]] #计算距离目标点向量差
        distiance_postiontotarget=np.linalg.norm(current_positiontotarget) #欧氏距离

        '''
        这里是把解耦的image于obv11进行处理得到最终的obv的代码
        '''

        observation={
            "image": image,   # (3, 224, 224), float32
            "vector": self.obv_normalize(Obv_11)     # (11,), float32
        }
        
        finshirate=Obv_11[10] #完成率

        # 用于计算是否前进了（未启用）
        if finshirate <= self.oldfinshirate:
            flag_dis = 0
        else:
            flag_dis = 1

        self.oldfinshirate = finshirate  # 更新上一次的完成率

        #再计算奖励
        reward = self.calculate_reward(Handdistance,0,update_flag,flag_dis,finshirate)
        print(f"reward: {reward}") #奖励值
        #判断是否游戏结束
        #done = self.is_done(update_flag)

        #terminated=done
        #truncated=done
        terminated = (update_flag == 1 or update_flag == 2)
        truncated = (self.current_step >= self.max_steps)


        self.olddistiance_positiontotarget=distiance_postiontotarget #数据更新
        return observation,reward,terminated,truncated,{}

    def is_done(self,gameover_flag):
        """
        判断游戏结束的标志
        """
        if gameover_flag == 2 or gameover_flag == 1 or self.current_step >= self.max_steps:
            return True
        else:
            return False

    def calculate_reward(self, postion_to_cl,curve,gameover_flag,flag_dis,finshirate):
        
        """
        计算奖励，自定义
        """
        #punishment_curve=0.5*curve_reward  # 弯曲奖励
        if curve<=0.005:
            punishment_curve=0
        elif curve<=0.15 and curve>0.005:
            punishment_curve=0.1
        else:
            punishment_curve=0.1

        punishment_cl=-0.1*postion_to_cl


        if gameover_flag == 2:
            reward_target=500
        elif gameover_flag == 1:
            reward_target=-50
            # reward_target=-50+finshirate*100
        else:
            reward_target=0
        #punishment_cl=0
        #reward_target=0
        #reward = reward_target+punishment_curve+punishment_cl+punishment_step+punishment_target
        reward = 0+0+punishment_cl+punishment_curve+flag_dis-0.1
        
        reward=(reward+reward_target)

        return reward

    def close(self):
        """
        关闭环境时需要清理的内容。
        """
        pass

    def obv_normalize(self,obv):
        """
        对obv进行归一化处理。
        """
        # 这里可以根据需要进行归一化处理
        # 例如：obv = (obv - min_obv) / (max_obv - min_obv)
        for i in range(6):
            obv[i]=obv[i]/100
        obv[9]=obv[9]/10
        return obv

    def _project_to_unit_circle(self, x, y):
        r = np.sqrt(x*x + y*y)
        if r > 1.0:
            x = x / r
            y = y / r
        return np.round(float(x), 3), np.round(float(y), 3)

