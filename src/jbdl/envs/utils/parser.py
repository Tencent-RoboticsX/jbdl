import os
import numpy as np
import pybullet
import gym
import gym.spaces
import gym.utils
from jbdl.envs import get_mjcf_path, get_urdf_path


class XmlBasedRobot:
    """
    Base class for mujoco .xml based agents.
    """

    self_collision = True

    def __init__(self, robot_name, action_dim, obs_dim, self_collision, add_ignored_joints=False):
        self.parts = None
        self.objects = []
        self.jdict = None
        self.ordered_joints = None
        self.robot_body = None
        self.add_ignored_joints = add_ignored_joints

        high = np.ones([action_dim])
        self.action_space = gym.spaces.Box(-high, high)
        high = np.inf * np.ones([obs_dim])
        self.observation_space = gym.spaces.Box(-high, high)

        self.robot_name = robot_name
        self.self_collision = self_collision

    def add_to_scene(self, bullet_client, bodies):
        self._p = bullet_client

        if self.parts is not None:
            parts = self.parts
        else:
            parts = {}

        if self.jdict is not None:
            joints = self.jdict
        else:
            joints = {}

        if self.ordered_joints is not None:
            ordered_joints = self.ordered_joints
        else:
            ordered_joints = []

        if np.isscalar(bodies):	 # streamline the case where bodies is actually just one body
            bodies = [bodies]

        dump = 0
        for i in range(len(bodies)):
            if self._p.getNumJoints(bodies[i]) == 0:
                part_name, robot_name = self._p.getBodyInfo(bodies[i])
                self.robot_name = robot_name.decode("utf8")
                part_name = part_name.decode("utf8")
                parts[part_name] = BodyPart(self._p, part_name, bodies, i, -1)
            for j in range(self._p.getNumJoints(bodies[i])):
                self._p.setJointMotorControl2(
                    bodies[i], j, pybullet.POSITION_CONTROL, positionGain=0.1, velocityGain=0.1, force=0)
                joint_info = self._p.getJointInfo(bodies[i], j)
                joint_name = joint_info[1]
                part_name = joint_info[12]

                joint_name = joint_name.decode("utf8")
                part_name = part_name.decode("utf8")

                if dump:
                    print("ROBOT PART '%s'" % part_name)
                if dump:
                    # limits = %+0.2f..%+0.2f effort=%0.3f speed=%0.3f" % ((joint_name,) + j.limits()) )
                    print("ROBOT JOINT '%s'" % joint_name)

                parts[part_name] = BodyPart(self._p, part_name, bodies, i, j)

                if part_name == self.robot_name:
                    self.robot_body = parts[part_name]

                if i == 0 and j == 0 and self.robot_body is None:  # if nothing else works, we take this as robot_body
                    parts[self.robot_name] = BodyPart(
                        self._p, self.robot_name, bodies, 0, -1)
                    self.robot_body = parts[self.robot_name]

                if joint_name[:6] == "ignore":
                    ignored_joint = Joint(self._p, joint_name, bodies, i, j)
                    ignored_joint.disable_motor()
                    # some of the robots (Hopper, Walker2D and HalfCheetah in mujoco)
                    # require read-access to these joints
                    if self.add_ignored_joints:
                        joints[joint_name] = ignored_joint
                        ordered_joints.append(ignored_joint)
                        joints[joint_name].power_coef = 0.0
                    continue

                if joint_name[:8] != "jointfix":
                    joints[joint_name] = Joint(
                        self._p, joint_name, bodies, i, j)
                    ordered_joints.append(joints[joint_name])

                    joints[joint_name].power_coef = 100.0

        return parts, joints, ordered_joints, self.robot_body

    def reset_pose(self, position, orientation):
        self.parts[self.robot_name].reset_pose(position, orientation)


class MJCFBasedRobot(XmlBasedRobot):
    """
    Base class for mujoco .xml based agents.
    """

    def __init__(self, model_xml, robot_name, action_dim, obs_dim, self_collision=True, add_ignored_joints=False):
        XmlBasedRobot.__init__(self, robot_name, action_dim,
                               obs_dim, self_collision, add_ignored_joints)
        self.model_xml = model_xml
        self.done_loading = 0

    def load(self, bullet_client):

        full_path = os.path.join(get_mjcf_path(), self.model_xml)

        self._p = bullet_client
        #print("Created bullet_client with id=", self._p._client)
        if self.done_loading == 0:
            self.ordered_joints = []
            self.done_loading = 1
            if self.self_collision:
                self.objects = self._p.loadMJCF(
                    full_path, flags=pybullet.URDF_USE_SELF_COLLISION
                    | pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
                self.parts, self.jdict, self.ordered_joints, self.robot_body = self.add_to_scene(
                    self._p, self.objects)
            else:
                self.objects = self._p.loadMJCF(full_path)
                self.parts, self.jdict, self.ordered_joints, self.robot_body = self.add_to_scene(
                    self._p, self.objects)


class URDFBasedRobot(XmlBasedRobot):
    """
    Base class for URDF .xml based robots.
    """

    def __init__(self, model_urdf, robot_name, action_dim, obs_dim,
                 base_position=None, base_orientation=None, fixed_base=False, self_collision=False):
        XmlBasedRobot.__init__(
            self, robot_name, action_dim, obs_dim, self_collision)

        self.model_urdf = model_urdf
        self.base_position = base_position if base_position is not None else [
            0, 0, 0]
        self.base_orientation = base_orientation if base_orientation is not None else [
            0, 0, 0, 1]
        self.fixed_base = fixed_base

    def load(self, bullet_client):
        self._p = bullet_client
        self.ordered_joints = []

        full_path = os.path.join(get_urdf_path(), self.model_urdf)

        if self.self_collision:
            self.parts, self.jdict, self.ordered_joints, self.robot_body = self.add_to_scene(self._p,
				self._p.loadURDF(full_path,
								basePosition=self.base_position,
								baseOrientation=self.base_orientation,
								useFixedBase=self.fixed_base,
								flags=pybullet.URDF_USE_SELF_COLLISION))
        else:
            self.parts, self.jdict, self.ordered_joints, self.robot_body = self.add_to_scene(self._p,
				self._p.loadURDF(full_path,
								basePosition=self.base_position,
								baseOrientation=self.base_orientation,
								useFixedBase=self.fixed_base))


class SDFBasedRobot(XmlBasedRobot):
    """
    Base class for SDF robots in a Scene.
    """

    def __init__(self, model_sdf, robot_name, action_dim, obs_dim,
                 basePosition=None, baseOrientation=None, fixed_base=False, self_collision=False):
        XmlBasedRobot.__init__(
            self, robot_name, action_dim, obs_dim, self_collision)

        if basePosition is None:
            basePosition = [0, 0, 0]
        if baseOrientation is None:
            baseOrientation = [0, 0, 0, 1]

        self.model_sdf = model_sdf
        self.fixed_base = fixed_base

    def reset(self, bullet_client):
        self._p = bullet_client

        self.ordered_joints = []

        self.parts, self.jdict, self.ordered_joints, self.robot_body = self.add_to_scene(self._p,
            self._p.loadSDF(os.path.join("models_robot", self.model_sdf)))
        # TODO: Not sure if this works, try it with kuka

        self.robot_specific_reset(self._p)

        # optimization: calc_state() can calculate something in self.* for calc_potential() to use
        s = self.calc_state()
        self.potential = self.calc_potential()

        return s

    def calc_potential(self):
        return 0


class PoseHelper:  # dummy class to comply to original interface
    def __init__(self, body_part):
        self.body_part = body_part

    def xyz(self):
        return self.body_part.current_position()

    def rpy(self):
        return pybullet.getEulerFromQuaternion(self.body_part.current_orientation())

    def orientation(self):
        return self.body_part.current_orientation()

    def speed(self):
        return self.body_part.speed()


class BodyPart:
    def __init__(self, bullet_client, body_name, bodies, body_index, body_part_index):
        self.bodies = bodies
        self._p = bullet_client
        self.body_index = body_index
        self.body_part_index = body_part_index
        self.initial_position = self.current_position()
        self.initial_orientation = self.current_orientation()
        self.bp_pose = PoseHelper(self)

    # a method you will most probably need a lot to get pose and orientation
    def state_fields_of_pose_of(self, body_id, link_id=-1):
        if link_id == -1:
            (x, y, z), (a, b, c, d) = self._p.getBasePositionAndOrientation(body_id)
        else:
            (x, y, z), (a, b, c, d), _, _, _, _ = self._p.getLinkState(
                body_id, link_id)
        return np.array([x, y, z, a, b, c, d])

    def get_pose(self):
        return self.state_fields_of_pose_of(self.bodies[self.body_index], self.body_part_index)

    def speed(self):
        if self.body_part_index == -1:
            (vx, vy, vz), _ = self._p.getBaseVelocity(
                self.bodies[self.body_index])
        else:
            (_, _, _), (_, _, _, _), _, _, _, _, (vx, vy, vz), (_, _, vy) = self._p.getLinkState(
                self.bodies[self.body_index], self.body_part_index, computeLinkVelocity=1)
        return np.array([vx, vy, vz])

    def current_position(self):
        return self.get_pose()[:3]

    def current_orientation(self):
        return self.get_pose()[3:]

    def get_position(self):
        return self.current_position()

    def get_orientation(self):
        return self.current_orientation()

    def get_velocity(self):
        return self._p.getBaseVelocity(self.bodies[self.body_index])

    def reset_position(self, position):
        self._p.resetBasePositionAndOrientation(
            self.bodies[self.body_index], position, self.get_orientation())

    def reset_orientation(self, orientation):
        self._p.resetBasePositionAndOrientation(
            self.bodies[self.body_index], self.get_position(), orientation)

    def reset_velocity(self, linear_velocity=None, angular_velocity=None):
        if linear_velocity is None:
            linear_velocity = [0, 0, 0]
        if angular_velocity is None:
            angular_velocity = [0, 0, 0]
        self._p.resetBaseVelocity(
            self.bodies[self.body_index], linear_velocity, angular_velocity)

    def reset_pose(self, position, orientation):
        self._p.resetBasePositionAndOrientation(
            self.bodies[self.body_index], position, orientation)

    def pose(self):
        return self.bp_pose

    def contact_list(self):
        return self._p.getContactPoints(self.bodies[self.body_index], -1, self.body_part_index, -1)


class Joint:
    def __init__(self, bullet_client, joint_name, bodies, body_index, joint_index):
        self.bodies = bodies
        self._p = bullet_client
        self.body_index = body_index
        self.joint_index = joint_index
        self.joint_name = joint_name

        joint_info = self._p.getJointInfo(
            self.bodies[self.body_index], self.joint_index)
        self.joint_type = joint_info[2]
        self.lower_limit = joint_info[8]
        self.upper_limit = joint_info[9]
        self.joint_has_limits = self.lower_limit < self.upper_limit
        self.joint_max_velocity = joint_info[11]
        self.power_coeff = 0

    def set_state(self, x, vx):
        self._p.resetJointState(
            self.bodies[self.body_index], self.joint_index, x, vx)

    def current_position(self):  # just some synonym method
        return self.get_state()

    def current_relative_position(self):
        pos, vel = self.get_state()
        if self.joint_has_limits:
            pos_mid = 0.5 * (self.lower_limit + self.upper_limit)
            pos = 2 * (pos - pos_mid) / (self.upper_limit - self.lower_limit)

        if self.joint_max_velocity > 0:
            vel /= self.joint_max_velocity
        elif self.joint_type == 0:  # JOINT_REVOLUTE_TYPE
            vel *= 0.1
        else:
            vel *= 0.5
        return (
            pos,
            vel
        )

    def get_state(self):
        x, vx, _, _ = self._p.getJointState(
            self.bodies[self.body_index], self.joint_index)
        return x, vx

    def get_position(self):
        x, _ = self.get_state()
        return x

    def get_orientation(self):
        _, r = self.get_state()
        return r

    def get_velocity(self):
        _, vx = self.get_state()
        return vx

    def set_position(self, position):
        self._p.setJointMotorControl2(
            self.bodies[self.body_index], self.joint_index, pybullet.POSITION_CONTROL, targetPosition=position)

    def set_velocity(self, velocity):
        self._p.setJointMotorControl2(
            self.bodies[self.body_index], self.joint_index, pybullet.VELOCITY_CONTROL, targetVelocity=velocity)

    def set_motor_torque(self, torque):  # just some synonym method
        self.set_torque(torque)

    def set_torque(self, torque):
        self._p.setJointMotorControl2(
			bodyIndex=self.bodies[self.body_index], jointIndex=self.joint_index,
            controlMode=pybullet.TORQUE_CONTROL, force=torque)  # , positionGain=0.1, velocityGain=0.1)

    def reset_current_position(self, position, velocity):  # just some synonym method
        self.reset_position(position, velocity)

    def reset_position(self, position, velocity):
        self._p.resetJointState(
            self.bodies[self.body_index], self.joint_index, targetValue=position, targetVelocity=velocity)
        self.disable_motor()

    def disable_motor(self):
        self._p.setJointMotorControl2(
			self.bodies[self.body_index], self.joint_index, controlMode=pybullet.POSITION_CONTROL,
            targetPosition=0, targetVelocity=0, positionGain=0.1, velocityGain=0.1, force=0)
