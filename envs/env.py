import time

import gym
import numpy as np
import pybullet as p
import pybullet_data as pd
from scipy.interpolate import interp1d
from envs.camera import Camera


URDF_FILENAME = "data/urdf/new2.urdf"
DEFAULT_POSITION = [0, 0, 0.73]
INIT_MOTOR_POS = [0., -2.6, -2.6, 0.]
INIT_EE_POS = np.array([0.1114, 0.06746, 1.2544])


def get_trajectory(tx, dt, spline=interp1d):
    t = tx[:, 0]
    ts = np.arange(t[0], t[-1], dt)
    trajectories = []
    for i in range(1, tx.shape[1]):
        tg = spline(t, tx[:, i])
        ti = tg(ts)
        trajectories.append(ti)
    return ts, np.array(trajectories).T


class ArmScanGym(gym.Env):
    def __init__(self, target_info=None, is_render=False, is_train=True):
        """
        :param target_info:
            (urdf_filename, target_initial_position, target_length_x, target_length_y)
        """
        self.target_info = target_info
        self.bullet_client = p.connect(p.GUI if is_render
                                       else p.DIRECT)
        self.is_render = is_render
        self.is_train = is_train
        self.dt = 0.1
        self._build_sim()

        # joint min: [-np.pi, -np.pi, -2.9, -2.]
        # joint max: [np.pi, 0., 2.9, 2.]
        self.action_space = gym.spaces.Box(-1., 1., shape=(4,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-1., 1., shape=(17,), dtype=np.float32)

        self.obstacle_urdf = "data/urdf/L.urdf"
        self.obstacle = None  # 在 reset 方法中初始化
        self.prev_action = None

        self.reset_time = 0

        if not is_train:
            self.camera = Camera(INIT_EE_POS, p.getQuaternionFromEuler([0, -np.pi / 2, 0.]))

    def get_obs(self):
        ee_pose = self.ee_pose()
        tar_pos = np.copy(self.target_ee_position)
        tar_ee_pos = tar_pos - ee_pose[0]
        joint_pos = self.get_joint_position()
        joint_vel = self.get_joint_velocity()
        obs = np.hstack([
            ee_pose[0],
            ee_pose[1],
            tar_ee_pos,
            joint_pos / (np.pi * 2),
            joint_vel
        ]).astype(np.float32)
        return obs

    def reset(self):
        time_interval = 3

        for i in range(len(self.jointIds)):
            p.resetJointState(self.robot_body, self.jointIds[i], INIT_MOTOR_POS[i])

        if self.is_train:
            surface = np.random.uniform(np.float32([0.4, 0.4, 1.8]),
                                        np.float32([1.2, 1.2, 1.9]))
        else:
            surface = self.target_info[1]

        target_key_points = np.array([
            [0 * time_interval, INIT_EE_POS[0], INIT_EE_POS[1], INIT_EE_POS[2]],
            [1 * time_interval, surface[0] / 2, -surface[1] / 2, surface[2]],
            [2 * time_interval, surface[0] / 2, surface[1] / 2, surface[2]],
            [3 * time_interval, -surface[0] / 2, surface[1] / 2, surface[2]],
            [4 * time_interval, -surface[0] / 2, -surface[1] / 2, surface[2]],
            [5 * time_interval, INIT_EE_POS[0], INIT_EE_POS[1], INIT_EE_POS[2]],
        ])
        ts, self.trajectory = get_trajectory(target_key_points, 0.1)
        # 移除原有的障碍物（如果存在）
        if self.obstacle is not None:
            p.removeBody(self.obstacle)

        # 随机生成新的障碍物位置
        obstacle_pos = np.random.uniform(low=[0.2, 0, 1.4], high=[0.6, 0.5, 1.7])

        # 加载新的障碍物（设置为固定基座以防止掉落）
        self.obstacle = p.loadURDF(self.obstacle_urdf, obstacle_pos, useFixedBase=True)
        self.max_episode_len = len(self.trajectory)
        self.target_ee_position = INIT_EE_POS
        self.reset_time += 1
        self.time_step = 0

        return self.get_obs()

    def step(self, action):
        target_pos = self.get_joint_position() + action * (np.pi / 20)
        for _ in range(10):
            p.setJointMotorControlArray(self.robot_body,
                                        self.jointIds,
                                        p.POSITION_CONTROL,
                                        target_pos)
            p.stepSimulation()
            if self.is_render:
                time.sleep(0.01)
                cam_pose = self.get_camera_pose()
                self.image = self.camera.get_image()
                self.camera.update_camera_pose(cam_pose[0] + np.array([0., 0., 0.1]),
                                               p.getQuaternionFromEuler(
                                                   p.getEulerFromQuaternion(cam_pose[1]) + np.array(
                                                       [0., -np.pi / 2, 0])))
        reward = 0  # 初始化奖励

        curr_dist = np.linalg.norm(self.target_ee_position - self.ee_pose()[0])
        pos_reward = np.exp(-10 * np.abs(curr_dist))  # (last_dist - curr_dist) * 50
        ori_reward = np.exp(-np.abs(self.ee_pose()[1][1] - (-np.pi / 2)))

        # 检查与障碍物的接触
        contact_with_obstacle = len(p.getContactPoints(self.robot_body, self.obstacle)) > 0

        # 如果与障碍物接触，给出负奖励
        obstacle_penalty = -1 if contact_with_obstacle else 0

        # 计算动作变化率的惩罚项（如果有上一个动作的话）
        if self.prev_action is not None:
            action_change = np.linalg.norm(action - self.prev_action)
            action_smoothness_penalty = -0.1 * action_change
        else:
            action_smoothness_penalty = 0.0

        self.prev_action = action  # 更新上一个动作

        reward = 5 * pos_reward + ori_reward + obstacle_penalty + action_smoothness_penalty

        contact = self.get_contact()

        done = contact or self.time_step >= self.max_episode_len - 2  #
        self.time_step += 1
        self.target_ee_position = self.trajectory[self.time_step]
        return self.get_obs(), reward, done, {}

    def render(self, mode="human"):
        pass

    def _build_sim(self):
        # basic
        p.resetSimulation()
        p.setAdditionalSearchPath(pd.getDataPath())
        p.loadURDF('plane_implicit.urdf')
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(0.01)
        p.resetDebugVisualizerCamera(cameraDistance=2.5,
                                     cameraYaw=-40,
                                     cameraPitch=-50,
                                     cameraTargetPosition=[0, 0, 0])
        self.robot_body = p.loadURDF(URDF_FILENAME, DEFAULT_POSITION, useFixedBase=True)



        self.jointIds = []
        self.link_name_dict = {}
        for j in range(p.getNumJoints(self.robot_body)):
            p.changeDynamics(self.robot_body, j, linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self.robot_body, j)
            jointType = info[2]
            link_name = info[-5].decode()
            self.link_name_dict[link_name] = j
            if info[1].decode() in ['joint2', 'joint3', 'joint4', 'joint5']:
                self.jointIds.append(j)
        for i in range(len(self.jointIds)):
            p.resetJointState(self.robot_body, self.jointIds[i], INIT_MOTOR_POS[i])

        if self.is_train:
            self.target = p.loadURDF("data/urdf/evn.urdf", [0, 0, 2.0], useFixedBase=True)
        else:
            self.target = p.loadURDF(self.target_info[0][0], self.target_info[0][1], useFixedBase=True)

    def is_success(self, pos_threshold=0.05):
        ee_pos = self.ee_pose()[0]
        tar_pos = np.copy(self.target_ee_position)
        curr_dist = np.linalg.norm(tar_pos - ee_pos)
        return curr_dist < pos_threshold

    def get_joint_position(self):
        return np.array([info[0] for info in p.getJointStates(self.robot_body, self.jointIds)])

    def get_joint_velocity(self):
        return np.array([info[1] for info in p.getJointStates(self.robot_body, self.jointIds)])

    def ee_pose(self):
        ee_state = p.getLinkState(self.robot_body, self.link_name_dict['link5'])
        return np.array(ee_state[0]), np.array(p.getEulerFromQuaternion(ee_state[1]))

    def get_camera_pose(self):
        cam_state = p.getLinkState(self.robot_body, self.link_name_dict['camera'])
        return np.array(cam_state[0]), np.array(cam_state[1])

    def get_contact(self):
        return len(p.getContactPoints(self.robot_body)) > 1


if __name__ == '__main__':
    env = ArmScanGym(target_info=(["data/urdf/evn.urdf", [0, 0, 2.0]], [1.2, 0.6, 1.7]), is_render=True)
    env.reset()

    for _ in range(10000):
        act = env.action_space.sample()
        env.step(act)
