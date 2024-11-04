import warnings

import pybullet as p
import numpy as np
import cv2


# cv2.cvtColor(np.zeros(shape), cv2.COLOR_RGB2BGR)
class ImgMap:
    def __init__(self,
                 shape,
                 world_limit,
                 back_color=(255, 255, 255)):
        assert len(shape) == 2
        assert np.shape(world_limit) == (2, 2)
        self.shape = np.array(shape)
        self.world_limit = np.array(world_limit)
        self._bg_img = np.zeros((shape[1], shape[0], 3))
        self._bg_img[:] = np.uint8(back_color)
        self.img = np.copy(self._bg_img)

    def to_img(self, pos):
        if np.size(pos) == 1:
            norm_pos = pos / np.subtract(np.mean(self.world_limit[1]),
                                         np.mean(self.world_limit[0]))
            return np.uint8(norm_pos * np.mean(self.shape))
        else:
            norm_pos = (pos - self.world_limit[0]) / np.subtract(self.world_limit[1], self.world_limit[0])
            img_x = np.int64(norm_pos * self.shape)
            return img_x

    def reset(self):
        self.img = np.copy(self._bg_img)

    def add_circle(self, pos, radius, color, thickness=-1):
        cv2.circle(self.img, self.to_img(pos), self.to_img(radius), color, thickness)

    def add_box(self, pos, width, color, thickness=-1):
        width = self.to_img(width)
        pos = self.to_img(pos)
        pt1 = pos - width / 2
        pt2 = pos + width / 2
        cv2.rectangle(self.img, np.int64(pt1), np.int64(pt2), color, thickness=thickness)

    def add_line(self, start, end, color, thickness=2):
        cv2.line(self.img, self.to_img(start), self.to_img(end), color, thickness=thickness)

    def large_img(self, size=(600, 600)):
        return cv2.resize(np.uint8(self.img), size, interpolation=cv2.INTER_NEAREST)

    def show(self, name="img"):
        cv2.imshow(name, cv2.cvtColor(np.uint8(self.img), cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) == ord('q'):
            # press q to terminate the loop
            cv2.destroyAllWindows()
        return self.img


def change_body_color(object_uid, rgba_color):
    p.changeVisualShape(object_uid, -1, rgbaColor=rgba_color)


def apply_texture(object_uid, texture_file):
    texUid = p.loadTexture(texture_file)
    p.changeVisualShape(object_uid, -1, textureUniqueId=texUid)


def create_obj(filename,
               position, orientation=(0, 0, 0, 1),
               is_collision=True, base_mass=0, rgba_color=(0.7, 0.4, 0, 1), mesh_scale=(1, 1, 1)):
    shape_parm = dict(shapeType=p.GEOM_MESH,
                      fileName=filename,
                      rgbaColor=rgba_color,
                      meshScale=mesh_scale)
    if len(orientation) == 3:
        orientation = p.getQuaternionFromEuler(orientation)
    body_parm = dict(baseMass=base_mass,
                     basePosition=position,
                     baseOrientation=orientation)
    return _create_body(shape_parm, body_parm, is_collision, p.GEOM_MESH)


def create_sphere(radius,
                  position, orientation=(0, 0, 0, 1),
                  is_collision=True, base_mass=0, rgba_color=(0.7, 0.4, 0, 1), mesh_scale=(1, 1, 1)):
    shape_parm = dict(shapeType=p.GEOM_SPHERE,
                      radius=radius,
                      rgbaColor=rgba_color,
                      meshScale=mesh_scale)
    if len(orientation) == 3:
        orientation = p.getQuaternionFromEuler(orientation)
    body_parm = dict(baseMass=base_mass,
                     basePosition=position,
                     baseOrientation=orientation)
    return _create_body(shape_parm, body_parm, is_collision, p.GEOM_SPHERE)


def create_box(half_extents,
               position, orientation=(0, 0, 0, 1),
               is_collision=True, base_mass=0, rgba_color=(0.7, 0.4, 0, 1), mesh_scale=(1, 1, 1)):
    assert len(half_extents) == 3
    shape_parm = dict(shapeType=p.GEOM_BOX,
                      halfExtents=half_extents,
                      rgbaColor=rgba_color,
                      meshScale=mesh_scale)
    if len(orientation) == 3:
        orientation = p.getQuaternionFromEuler(orientation)
    body_parm = dict(baseMass=base_mass,
                     basePosition=position,
                     baseOrientation=orientation)
    return _create_body(shape_parm, body_parm, is_collision, p.GEOM_BOX)


def create_cylinder(radius, length,
                    position, orientation=(0, 0, 0, 1),
                    is_collision=True, base_mass=0, rgba_color=(0.7, 0.4, 0, 1), mesh_scale=(1, 1, 1)):
    shape_parm = dict(shapeType=p.GEOM_CYLINDER,
                      radius=radius,
                      length=length,
                      rgbaColor=rgba_color,
                      meshScale=mesh_scale)
    if len(orientation) == 3:
        orientation = p.getQuaternionFromEuler(orientation)
    body_parm = dict(baseMass=base_mass,
                     basePosition=position,
                     baseOrientation=orientation)
    return _create_body(shape_parm, body_parm, is_collision, p.GEOM_CYLINDER)


def _create_body(shape_parm, body_parm, is_collision=True, shape_type=None):
    assert len(shape_parm["rgbaColor"]) == 4
    visualShapeId = p.createVisualShape(**shape_parm,
                                        specularColor=(0.4, .4, 0))
    if shape_type == p.GEOM_MESH:
        collisionShapeId = p.createCollisionShape(shapeType=shape_parm['shapeType'],
                                                  fileName=shape_parm['fileName'])
    elif shape_type == p.GEOM_SPHERE:
        collisionShapeId = p.createCollisionShape(shapeType=shape_parm['shapeType'],
                                                  radius=shape_parm['radius'])
    elif shape_type == p.GEOM_BOX:
        collisionShapeId = p.createCollisionShape(shapeType=shape_parm['shapeType'],
                                                  halfExtents=shape_parm['halfExtents'])
    elif shape_type == p.GEOM_CYLINDER or shape_type == p.GEOM_CAPSULE:
        collisionShapeId = p.createCollisionShape(shapeType=shape_parm['shapeType'],
                                                  radius=shape_parm['radius'],
                                                  height=shape_parm['length'])
    else:
        collisionShapeId = p.createCollisionShape(shapeType=shape_parm['shapeType'])

    if is_collision:
        body = p.createMultiBody(**body_parm,
                                 baseCollisionShapeIndex=collisionShapeId,
                                 baseVisualShapeIndex=visualShapeId)
    else:
        body = p.createMultiBody(**body_parm,
                                 baseVisualShapeIndex=visualShapeId)
    # set to default dynamics
    p.changeDynamics(body, -1, lateralFriction=0.5, spinningFriction=0.001, rollingFriction=0.001)
    return body


def generate_wave(freq, size):
    k_1 = np.random.uniform(0.8, 1), np.random.random() * 2 * np.pi
    k_2 = np.random.uniform(0.01, 1), np.random.random() * 2 * np.pi
    k_3 = np.random.uniform(0.01, 1), np.random.random() * 2 * np.pi
    A = lambda x: 0.3 * np.cos(freq * x + k_1[1])
    B = lambda x: 0.3 * np.cos(k_2[0] * freq * x + k_2[1])
    C = lambda x: 0.3 * np.cos(k_3[0] * freq * x + k_3[1])
    x1 = np.array([A(i) for i in range(size)])
    x2 = np.array([B(i) for i in range(size)])
    x3 = np.array([C(i) for i in range(size)])
    xs = x1 + x2 + x3
    return xs


CYLINDER = 1
SQUARE = 2


class PathTrack:
    def __init__(self, init_pos, rgb=(1., 0., 0.), width=1, trail_duration=0):
        self.init_pos = init_pos
        self.pos_history = [init_pos]
        self.trail_duration = trail_duration
        self.rgb = rgb
        self.width = width
        self.line = None

    def update(self, pos):
        self.pos_history.append(pos)
        p.addUserDebugLine(self.pos_history[-2],
                           self.pos_history[-1],
                           lineColorRGB=self.rgb,
                           lineWidth=self.width,
                           lifeTime=self.trail_duration)


class Space:
    """
    only consider 2D occupation
    """

    def __init__(self, low, high):
        self.low = np.array(low)
        self.high = np.array(high)
        self.none_spaces = []
        self.none_dots = []

    def copy(self):
        return Space(self.low, self.high)

    def sample(self, n=1, radius=None, trails=10000):
        points = []
        for i in range(n):
            for t in range(trails):
                pt = np.random.uniform(self.low, self.high)
                # check for occupation
                not_in_sp = not np.any([sp.contains(pt) for sp in self.none_spaces])
                far_from_dots = self.far_from_dots(pt, self.none_dots)
                if radius:
                    far_from_dots = far_from_dots and self.far_from_dots(pt, [[pp, radius] for pp in points])
                out = not_in_sp and far_from_dots
                if out:
                    break
                if t == trails - 1:
                    warnings.warn('exceed max pos finding num')
            points.append(pt)
        return np.array(points)

    def add_none_space(self, space):
        assert isinstance(space, Space)
        self.none_spaces.append(space)

    def add_none_dots(self, points, radius):
        if np.ndim(points) == 1:
            self.none_dots.append((points[:2], radius))
        else:
            for point in points:
                self.none_dots.append((point[:2], radius))

    def clear_all(self):
        self.none_dots = []
        self.none_spaces = []

    def contains(self, x):
        if isinstance(x, Space):
            return np.all(np.logical_and(self.low[:2] < x.low[:2], x.high[:2] < self.high[:2]))
        else:
            return np.all(np.logical_and(self.low[:2] < x[:2], x[:2] < self.high[:2]))

    def hard_contains(self, x):
        return np.all(np.logical_and(self.low < x, x < self.high))

    def far_from_dots(self, pt, dots):
        if len(dots) == 0:
            return True
        none_dots = np.array([d[0][:2] for d in dots])
        none_radius = np.array([d[1] for d in dots])
        none_dist = np.linalg.norm(none_dots - pt[:2], axis=1)
        return np.all(none_dist > none_radius)


class GridTerrain:
    def __init__(self,
                 land_size=(600, 600),
                 scale=0.05,
                 lateral_friction=0.9,
                 texture=None,
                 rgba_color=None):
        assert type(land_size[0]) == int and type(land_size[1]) == int
        self.scale = scale
        self.land_size = land_size
        self.lateral_friction = lateral_friction
        self.ground_id = None
        self._height_field = None
        self.texture = texture
        self.rgba_color = rgba_color

    def wave_generate(self, freq=0.06, max_height=1.2):
        x_wave = generate_wave(freq, self.land_size[0]).reshape(1, -1)
        y_wave = generate_wave(freq, self.land_size[1]).reshape(-1, 1)
        field = np.ones(self.land_size) * x_wave * y_wave * (max_height / self.scale)
        return field

    def get_height_field(self):
        height_field = self._height_field
        height_field[0, 0] = 0
        height_field[0, 1] = 0
        return height_field

    def get_height_from_position(self, position):
        x, y = np.array(position) // self.scale
        origin = int(self.land_size[0] / 2), int(self.land_size[1] / 2)
        return self.get_height_field()[origin[0] + int(x),
                                       origin[1] + int(y)]

    def build(self, height_field):
        self._height_field = height_field
        if self.ground_id:
            p.removeBody(self.ground_id)
        assert type(self._height_field) == np.ndarray
        center = (self.land_size[0] // 2, self.land_size[1] // 2)
        origin_height = np.mean(self._height_field[center[0] - 5:center[0] + 5, center[1] - 5: center[1] + 5])
        # self._height_field[0, 0] = -100
        # self._height_field[0, 1] = 100 - origin_height
        # prevent nan
        self._height_field[np.isnan(self._height_field)] = 0

        terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD,
                                              meshScale=(self.scale, self.scale, self.scale),
                                              heightfieldData=self._height_field.flatten(),
                                              numHeightfieldRows=self.land_size[0],
                                              numHeightfieldColumns=self.land_size[1])
        self.ground_id = p.createMultiBody(0, terrainShape)
        p.resetBasePositionAndOrientation(self.ground_id, (0, 0, 0), [0, 0, 0, 1])
        p.changeDynamics(self.ground_id, -1, lateralFriction=self.lateral_friction)
        if self.texture:
            apply_texture(self.ground_id, self.texture)
        if self.rgba_color:
            change_body_color(self.ground_id, self.rgba_color)


class Lidar:
    def __init__(self,
                 bullet_client,
                 body_id,
                 lidar_link_id=-1,
                 starting_rad=0.,
                 num_rays=180,
                 ray_length=(0.32, 2.),
                 render=True,
                 render_rate=0.1):
        self.p = bullet_client
        self.body_id = body_id
        self.lidar_link_id = lidar_link_id
        self.starting_rad = starting_rad
        self.num_rays = num_rays
        self.ray_length = ray_length
        self.render = render
        self.num_render_ray = np.ceil(num_rays * render_rate)
        self._build()
        self.update()

    def _build(self):
        replaceLines = True
        self.rayFrom = []
        self.rayTo = []
        self.rayIds = []
        self.rayHitColor = [1, 0, 0]
        self.rayMissColor = [0, 1, 0]
        for i in range(self.num_rays):
            theta = self.starting_rad + i * (np.pi * 2 / self.num_rays)
            self.rayFrom.append([self.ray_length[0] * np.sin(theta),
                                 self.ray_length[0] * np.cos(theta), 0])
            self.rayTo.append([self.ray_length[1] * np.sin(theta),
                               self.ray_length[1] * np.cos(theta), 0])
            if replaceLines and self.render and i % (self.num_rays // self.num_render_ray) == 0:
                self.rayIds.append(
                    self.p.addUserDebugLine(self.rayFrom[i],
                                            self.rayTo[i],
                                            self.rayMissColor,
                                            parentObjectUniqueId=self.body_id,
                                            parentLinkIndex=self.lidar_link_id))
            else:
                self.rayIds.append(-1)

    def update(self):
        self.lidar_data = []
        numThreads = 0
        result = self.p.rayTestBatch(self.rayFrom,
                                     self.rayTo,
                                     numThreads,
                                     parentObjectUniqueId=self.body_id,
                                     parentLinkIndex=self.lidar_link_id)

        for i in range(self.num_rays):
            # hitObjectUid = result[i][0]
            hitFraction = result[i][2]
            fraction_dis = self.ray_length[1] * hitFraction
            self.lidar_data.append(fraction_dis)
            # hitPosition = result[i][3]
            if self.render:
                if i % (self.num_rays // self.num_render_ray) == 0:
                    if hitFraction == 1.:
                        self.p.addUserDebugLine(self.rayFrom[i],
                                                self.rayTo[i],
                                                self.rayMissColor,
                                                replaceItemUniqueId=self.rayIds[i],
                                                parentObjectUniqueId=self.body_id,
                                                parentLinkIndex=self.lidar_link_id)
                    else:
                        localHitTo = [self.rayFrom[i][0] + hitFraction * (self.rayTo[i][0] - self.rayFrom[i][0]),
                                      self.rayFrom[i][1] + hitFraction * (self.rayTo[i][1] - self.rayFrom[i][1]),
                                      self.rayFrom[i][2] + hitFraction * (self.rayTo[i][2] - self.rayFrom[i][2])]
                        self.p.addUserDebugLine(self.rayFrom[i],
                                                localHitTo,
                                                self.rayHitColor,
                                                replaceItemUniqueId=self.rayIds[i],
                                                parentObjectUniqueId=self.body_id,
                                                parentLinkIndex=self.lidar_link_id)
                        # p.removeUserDebugItem()

    def get_data(self):
        return self.lidar_data

    def reset(self):
        self.p.removeAllUserDebugItems()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    import pybullet_data

    # p.connect(p.GUI)
    # g = GridTerrain()
    # for _ in range(30):
    #     g.add_object(CYLINDER, np.random.randint((-600, -600), (600, 600)), 1, np.random.random() * 2)
    #     g.add_object(SQUARE, np.random.randint((-600, -600), (600, 600)), 1, np.random.random() * 2)
    # g.build()
    # lp = PathTrack((0, 0, 0.5))
    # for i in range(1000):
    #     p.stepSimulation()
    #     time.sleep(0.1)
    #     lp.update((np.cos(i * 0.1) * 0.8 + (i * 0.01), np.sin(i * 0.1) * 0.3, 0.5))

    sp0 = Space(np.array([-1, -1]), np.array([1, 1]))
    sp1 = Space(np.array([-0.5, -0.5]), np.array([.5, .5]))
    sp2 = Space(np.array([-1, -1]), np.array([1, 1]))

    samples2 = sp2.sample(10)

    sp0.add_none_space(sp1)
    sp0.add_none_dots(samples2, radius=0.1)

    samples0 = np.array(sp0.sample(1000))
    samples1 = np.array(sp1.sample(10, .1))

    plt.scatter(samples2[:, 0], samples2[:, 1])
    plt.scatter(samples0[:, 0], samples0[:, 1])
    plt.scatter(samples1[:, 0], samples1[:, 1])
    plt.show()
