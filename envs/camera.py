"""Simulated cameras for rendering images."""
import itertools
import typing
import pathlib

import yaml
import numpy as np
import pybullet
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt


class BaseCamera:
    def get_image(
            self, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
    ) -> np.ndarray:
        raise NotImplementedError()


class Camera(BaseCamera):


    def __init__(
            self,
            camera_position,
            camera_orientation,
            image_size=(270, 270),
            field_of_view=52,
            near_plane_distance=0.001,
            far_plane_distance=100.0,
            pybullet_client_id=0,
    ):

        self._pybullet_client_id = pybullet_client_id
        self._width = image_size[0]
        self._height = image_size[1]
        self._near = near_plane_distance
        self._far = far_plane_distance
        self.field_of_view = field_of_view
        self.update_camera_pose(camera_position, camera_orientation)


    def update_camera_pose(self, position, orientation):
        self.base_pose = (position, orientation)
        baseOrientation = orientation
        matrix = pybullet.getMatrixFromQuaternion(baseOrientation,
                                                  physicsClientId=self._pybullet_client_id)
        tx_vec = np.array([matrix[0], matrix[3], matrix[6]])
        tz_vec = np.array([matrix[2], matrix[5], matrix[8]])
        cameraPos = np.array(position)
        targetPos = cameraPos + 1 * tx_vec

        self._view_matrix = pybullet.computeViewMatrix(
            cameraEyePosition=cameraPos,
            cameraTargetPosition=targetPos,
            cameraUpVector=tz_vec,
            physicsClientId=self._pybullet_client_id
        )
        self._proj_matrix = pybullet.computeProjectionMatrixFOV(
            fov=self.field_of_view,
            aspect=float(self._width) / self._height,
            nearVal=self._near,
            farVal=self._far,
            physicsClientId=self._pybullet_client_id
        )

    def get_image(
            self, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
    ) -> np.ndarray:

        (_, _, img, _, _) = pybullet.getCameraImage(
            width=self._width,
            height=self._height,
            viewMatrix=self._view_matrix,
            projectionMatrix=self._proj_matrix,
            renderer=renderer,
            physicsClientId=self._pybullet_client_id,
        )
        # remove the alpha channel
        return img[:, :, :3]

    def get_depth_image(
            self, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
    ) -> np.ndarray:

        (_, _, _, depthImg, _) = pybullet.getCameraImage(
            width=self._width,
            height=self._height,
            viewMatrix=self._view_matrix,
            projectionMatrix=self._proj_matrix,
            renderer=renderer,
            physicsClientId=self._pybullet_client_id,
        )
        return depthImg

    def get_depth_map(
            self, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
    ) -> np.ndarray:

        (_, _, _, depthImg, _) = pybullet.getCameraImage(
            width=self._width,
            height=self._height,
            viewMatrix=self._view_matrix,
            projectionMatrix=self._proj_matrix,
            renderer=renderer,
            physicsClientId=self._pybullet_client_id,
        )
        # remove the alpha channel

        depthMap = self._far * self._near / (self._far - (self._far - self._near) * depthImg)
        return depthMap

    def get_object_position(self, pixel_position):
        h, w = pixel_position
        projectionMatrix = np.array(self._proj_matrix).reshape((4, 4), order='F')
        viewMatrix = np.array(self._view_matrix).reshape((4, 4), order='F')
        tran_pix_world = np.linalg.inv(np.matmul(projectionMatrix, viewMatrix))
        x = 2 * w / self._width - 1.
        y = 1. - 2 * h / self._height  # be careful！ depth and its corresponding position
        z = 2 * self.get_depth_image()[h, w] - 1
        pixPos = np.asarray([x, y, z, 1])
        position = np.matmul(tran_pix_world, pixPos)
        object_position = position[:3] / position[3]
        return object_position - self.base_pose[0]

    def plot_point_cloud(self):
        import OpenGL.GL as gl
        import OpenGL.GLU as glu
        # plot
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        stepX = 30
        stepY = 30
        pointCloud = np.empty([np.int(self._height / stepY), np.int(self._width / stepX), 4])
        projectionMatrix = np.asarray(self._proj_matrix).reshape([4, 4], order='F')
        viewMatrix = np.asarray(self._view_matrix).reshape([4, 4], order='F')
        tran_pix_world = np.linalg.inv(np.matmul(projectionMatrix, viewMatrix))
        for h in range(0, self._height, stepY):
            for w in range(0, self._width, stepX):
                x = 2 * w / self._width - 1.
                y = 1. - 2 * h / self._height  # be careful！ depth and its corresponding position
                z = 2 * self.get_depth_image()[h, w] - 1
                pixPos = np.asarray([x, y, z, 1])
                position = np.matmul(tran_pix_world, pixPos)
                point = position[:3] / position[3]
                pointCloud[np.int(h / stepY), np.int(w / stepX), :] = point
                ax.scatter(point[0], point[1], point[2])
        plt.show()
