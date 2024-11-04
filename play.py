import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.results_plotter import load_results
from algo.sac import SAC

from envs.env import ArmScanGym
from algo.ddpg import DDPG
from algo.td3 import TD3

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

RESULT_DIR = "./result"

end_effector_positions = []

def window_smooth(v, window_size):
    length = int(len(v) // window_size)
    windowed = np.array(v[:length * window_size]).reshape(-1, window_size).mean(1)
    return windowed


def plot_results(logdir, num=100):
    os.makedirs(RESULT_DIR, exist_ok=True)
    list_dir = os.listdir(logdir)
    train_steps = [i * 1e6 / num for i in range(num)]

    plt.ylabel('Returns')
    plt.xlabel('Train Steps')


    for log_folder in list_dir:
        result = load_results(f"{logdir}/{log_folder}")
        R = result['r']
        filtered = window_smooth(R, len(R) // num)
        plt.plot(train_steps, filtered, label=log_folder)
    plt.grid()
    plt.legend()
    plt.savefig(f"{RESULT_DIR}/return.png")
    plt.show()


def play(logdir):
    import pybullet as p
    env = ArmScanGym(target_info=(["data/urdf/evn.urdf", [0, 0, 2.0]], [1.2, 0.5, 1.7]),
                     is_render=True, is_train=False)
    model = SAC.load(f"{logdir}/sac/best_model")
    print(model.policy)
    obs = env.reset()

    reward = 0
    # Initialize a list to store end-effector positions
    end_effector_positions = []
    for i in range(15000):
        act = model.predict(obs, deterministic=True)[0]
        obs, r, d, _ = env.step(act)

        current_ee_pose = env.ee_pose()  # Step 2
        end_effector_positions.append(current_ee_pose[0])

        reward += r
        if d:
            print(reward)
            env.reset()
            reward = 0

    end_effector_positions = np.array(end_effector_positions)

    # Create a new figure
    fig = plt.figure()

    # Add 3D subplot
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for 3D trajectory
    ax.scatter(end_effector_positions[:, 0], end_effector_positions[:, 1], end_effector_positions[:, 2])

    # Adding labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D End-Effector Trajectory')

    # Save the figure
    plt.savefig('9.14.1_3D_end_effector_trajectory.png')

    # Show the plot
    plt.show()



if __name__ == '__main__':
    # plot_results(logdir="./logs")
    play(logdir="./logs")

