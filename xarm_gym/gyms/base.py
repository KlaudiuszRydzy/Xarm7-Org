from abc import ABC, abstractmethod

import gymnasium as gym
from gymnasium import env
from misc.boundary import BoundaryManager
from misc.camera import Camera
from gello.cameras.realsense_camera import RealSenseCamera, get_device_ids
from xarm.wrapper import XArmAPI
from gello.data_utils.keyboard_interface import KBReset
# from misc.robot import XarmRobot


class LUCBase(gym.Env, ABC):
    """Base class for all LUC environments."""

    @abstractmethod
    def __init__(
        self, robot: XarmRobot, boundary: BoundaryManager, cameras: List[Camera]
    ):
        super().__init__()

        # set the boundaries
        # set the initial position
        # set the task description
        self.robot.set_collision_rebound(self, True) # rebound on collision

        # initialize connection to the robot
        # initialize the cameras

        # anything else?

    @abstractmethod
    def reset(self):
        pass

    def step(self, action):
        result = self.safety_check(self.position, action)
        if result is not None:
            return result
        else:
            return self._step(action)

    @abstractmethod
    def _step(self, action):
        pass

    @abstractmethod
    def render(self, mode="human"):
        """return the camera images from all cameras
        either cv2.imshow for 0.1 seconds or return the images as numpy arrays
        """

        if mode == "human":
            pass
        elif mode == "rgb_array":
            pass

    def safety_check(self, position, action):
        if not self.boundary.contains(self.position + action):
            # end the episode as a failure
            return self._state, -1, True, False, {}

    @property
    def position(self):
        return self.robot.position

    @property
    def angles(self):
        return self.robot.angles

    def set_mode(self, mode: int = 2):
        """xArm mode, only available in socket way and  enable_report is True

        :return:
            0: position control mode
            1: servo motion mode
            2: joint teaching mode
            3: cartesian teaching mode (invalid)
            4: joint velocity control mode
            5: cartesian velocity control mode
            6: joint online trajectory planning mode
            7: cartesian online trajectory planning mode
        """
        self.robot.set_mode(mode)

    @property
    def state(self):
        return {
            'cartesian': {
                "cartesian": self.robot.get_position_aa()[1][:3]
                'aa': self.robot.get_position_aa()[1][3:]/1000, # m to mm
                },
            "angles": self.robot.angles,
            'joint': {
                'position': [self.robot.get_joint_states(num=i)[1] for i in range(7)],
                'velocity': self.robot.realtime_joint_speeds, # deg/s or rad/s
                'torque': self.robot.joints_torque,
            },
            'speed': self.robot.last_used_joint_speed,
            'acceleration': self.robot.last_used_joint_acc,
            'temperature': self.robot.temperatures,
            'voltage': self.robot.voltages,
            'current': self.robot.currents,
            'is_moving': self.robot.is_moving,
            'gripper': {
                'position': self.robot.get_gripper_position,
                },
        }


    def set_position(self, position, wait=True, action_space='aa'):
        if action_space == 'aa':
            self.robot.set_position_aa(position, wait)
        elif action_space == 'joint':
            self.robot.set_joint_position(position, wait)
        else:
            raise ValueError(f"Invalid action space: {action_space}")

    def set_gripper(self, pos, wait=True):
        self.robot.set_gripper_position(pos, wait)

    def clean(self):
        # only use if specific error @ klaud figure out what these should be
        """
        Clean the error, need to be manually enabled 
        motion(arm.motion_enable(True)) and set state(arm.set_state(state=0))
        after clean error

        :return: code
            code: See the [API Code Documentation](./xarm_api_code.md#api-code) 
            for details.
        """
        self.robot.clean_error()
        self.robot.clean_gripper_error()
        self.robot.clean_warn()
        # @klaud do some reading to understand if robot.clean_error() is preferred
        # over robot.reset(params)

        # self.cameras.clean()
        # self.boundary.clean()
        # anything else?

    def kin_inv(self, pose, in_rad=None, out_rad=None):
        code, angles =  self.robot.get_inverse_kinematics(pose, in_rad, out_rad)
        if code == 0:
            return angles
        else:
            raise ValueError(f"Error in inverse kinematics: {code}")

    def kin_fwd(self, angles, in_rad=None, out_rad=None):
        code, pose = self.robot.get_forward_kinematics(angles, in_rad, out_rad)
        if code == 0:
            return pose
        else:
            raise ValueError(f"Error in inverse kinematics: {code}")

    def stop(self):
        try: 
            self.robot.set_mode(2)
        except:
            self.robot.emergency_stop()
            raise ValueError("Emergency stop activated")


