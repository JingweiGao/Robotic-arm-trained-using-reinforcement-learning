import time
import numpy as np

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from algo.sac import SAC
from algo.ddpg import DDPG
from algo.td3 import TD3
from algo.utils import SaveOnBestTrainingRewardCallback
from envs.env import ArmScanGym


def train_sac(env, log_dir="./logs/sac"):
    start_time = time.time()
    env = Monitor(env, f"{log_dir}/monitor.csv")
    model = SAC("MlpPolicy",
                env,
                learning_rate=1e-3,
                buffer_size=1_000_000,
                learning_starts=100,
                batch_size=100,
                tau=0.005,
                gamma=0.99,
                device='cuda:0',
                verbose=1)
    callback = SaveOnBestTrainingRewardCallback(int(1e4), log_dir)
    model.learn(total_timesteps=int(2e6), callback=callback)
    model.save(f"{log_dir}/last_model.zip")
    print(f"total train time: {time.time() - start_time}")


def train_ddpg(env, log_dir="./logs/ddpg"):
    start_time = time.time()
    env = Monitor(env, f"{log_dir}/monitor.csv")
    model = DDPG("MlpPolicy",
                 env,
                 learning_rate=1e-3,
                 buffer_size=1_000_000,
                 learning_starts=100,
                 batch_size=100,
                 tau=0.005,
                 gamma=0.99,
                 action_noise=NormalActionNoise(np.zeros(env.action_space.shape[0]),
                                                np.ones(env.action_space.shape[0]) * 0.1),
                 device='cuda:0',
                 verbose=1)
    callback = SaveOnBestTrainingRewardCallback(int(1e4), log_dir)
    model.learn(total_timesteps=int(2e6), callback=callback)
    model.save(f"{log_dir}/last_model.zip")
    print(f"total train time: {time.time() - start_time}")


def train_td3(env, log_dir="./logs/td3"):
    start_time = time.time()
    env = Monitor(env, f"{log_dir}/monitor.csv")
    model = TD3("MlpPolicy",
                env,
                learning_rate=1e-3,
                buffer_size=1_000_000,
                learning_starts=100,
                batch_size=100,
                tau=0.005,
                gamma=0.99,
                action_noise=NormalActionNoise(np.zeros(env.action_space.shape[0]),
                                               np.ones(env.action_space.shape[0]) * 0.1),
                device='cuda:0',
                verbose=1)
    callback = SaveOnBestTrainingRewardCallback(int(1e4), log_dir)
    model.learn(total_timesteps=int(2e6), callback=callback)
    model.save(f"{log_dir}/last_model.zip")
    print(f"total train time: {time.time() - start_time}")


if __name__ == '__main__':
    train_env = ArmScanGym()
    train_sac(train_env)
    # train_ddpg(train_env)
    # train_td3(train_env)
