import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# 全连接网络(弃用)
class CustomMLP(BaseFeaturesExtractor):
    def __init__(self,  observation_space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        self.mlp = nn.Sequential(
            nn.Linear(observation_space.shape[0],512),
            nn.Tanh(),
            nn.Linear(512,512),
            nn.Tanh(),
            nn.Linear(512,512),
            nn.Tanh(),
            nn.Linear(512,512),
            nn.Tanh(),
            nn.Linear(512, features_dim),
            nn.Tanh()
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.mlp(x)
#卷积网络（弃用）
class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self,  observation_space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        # 输入通道数

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 输出: (112, 112, 32)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 输出: (56, 56, 32)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 输出: (28, 28, 64)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 输出: (14, 14, 64)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 输出: (14, 14, 128)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7,7)),
            nn.Flatten()  # 输出: 14*14*128=25088
        )

        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
    
policy_kwargs = dict(
    features_extractor_class=CustomMLP, #全连接网络
    features_extractor_kwargs=dict(features_dim=512), #observation_space会自动传入在使用的时候
    net_arch=dict(pi=[512,512,512,512], vf=[512,512,512,512]),  # 策略网络和价值网络的隐藏层
    activation_fn=nn.Tanh  # 添加激活函数的定义
)

# 多模态输入（启用）
from gymnasium.spaces import Dict

class DualInputExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Dict, features_dim: int = 1024):
        super().__init__(observation_space, features_dim)

        # === 图像输入 shape: (3, 224, 224) ===
        img_shape = observation_space["image"].shape  # (3, 224, 224)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 输出: (112, 112, 32)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 输出: (56, 56, 32)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 输出: (28, 28, 64)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 输出: (14, 14, 64)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 输出: (14, 14, 128)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7,7)),
            nn.Flatten()  # 输出: 14*14*128=25088
        )

        # 动态计算 CNN 输出维度 理论上固定的
        with th.no_grad():
            dummy_input = th.zeros(1, *img_shape)
            cnn_output_dim = self.cnn(dummy_input).shape[1]

        self.image_fc = nn.Sequential(
            nn.Linear(cnn_output_dim, 512),
            nn.ReLU()
        )

        # === 向量输入 shape: (11,) ===
        sensor_dim = observation_space["vector"].shape[0]
        self.vector_fc = nn.Sequential(
            nn.Linear(sensor_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),  # 新增 Dropout 层
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),  # 新增 Dropout 层
            nn.Linear(512, 512),
            nn.ReLU()
        )

        # === 融合层输出为 1024维 ===
        self._features_dim = 1024  # 最终返回的 fused 向量维度

    def forward(self, observations):
        # 图像：标准化到 [0, 1]
        x_img = observations["image"]
        x_vec = observations["vector"]

        img_feat = self.image_fc(self.cnn(x_img))
        vec_feat = self.vector_fc(x_vec)

        fused = th.cat([img_feat, vec_feat], dim=1)  # (batch, 1024)
        return fused
    
policy_kwargs_Mlt = dict(
    features_extractor_class=DualInputExtractor,
    features_extractor_kwargs=dict(features_dim=1024),
    net_arch=dict(
        pi=[512,512],  # 策略网络的结构
        vf=[512,512]   # 价值网络的结构
    ),
    activation_fn=nn.ReLU
)   


