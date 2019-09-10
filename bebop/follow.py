# Geometrical entities in this module are specified in the body frame of the drone

import collections

import enum
import numpy as np

# import keras
# from keras import backend as K
# import tensorflow as tf

import torch
import sys
#sys.path.append("../../PyTorch")
from ModelTrainer import ModelTrainer
from ModelManager import ModelManager
from FrontNet import PreActBlock
from FrontNet import FrontNet
from geometry_msgs.msg import PoseStamped, Quaternion
from tf.transformations import quaternion_from_euler
import rospy



class Controller(enum.Enum):
    idle = 0
    exact = 1
    v1 = 2
    v2 = 3
    v3 = 4


# velocity is a tuple of three floats
# angular_speed is a float
# Example output = ControlOutput(velocity=(0.2, 3.2, 0), angular_speed=0.3)
ControlOutput = collections.namedtuple('ControlOutput', ['velocity', 'angular_speed'])


def _cmd_from_angular_speed(omega):
    return min(max(omega / 1.75, -1), 1)


def _clamp(xs, bss):
    return [max(min(x, bs[1]), bs[0]) for x, bs in zip(xs, bss)]

# TODO: Dario used delay 0.1 or 0 in training?
# TODO: Chack all other default params, should be the same as in training


def _controller(target_position, target_yaw, drone_velocity, velocity_head=[0, 0, 0], distance=1.5,
                delta_altitude=0.0, delay=0.0, tau=0.5, eta=1.0, rotation_tau=0.5, max_speed=1.5,
                max_ang_speed=2.0, max_acc=1.0, F=2.83):
    position_target = (np.array(target_position) +
                       np.array([np.cos(target_yaw) * distance, np.sin(target_yaw) * distance,
                                 delta_altitude]))
    des_drone_velocity = (np.array(position_target) - delay *
                          np.array(drone_velocity)) / eta + np.array(velocity_head)
    cmd_linear_z = des_drone_velocity[2]
    s_des = np.linalg.norm(des_drone_velocity[:2])
    if s_des > max_speed:
        des_drone_velocity = des_drone_velocity / s_des * max_speed
    des_horizontal_acceleration_drone = (
        des_drone_velocity[:2] - np.array(drone_velocity)[:2]) / tau
    des_horizontal_acceleration_drone = np.array(_clamp(
        des_horizontal_acceleration_drone, ((-max_acc, max_acc), (-max_acc, max_acc))))
    cmd_linear_x, cmd_linear_y = des_horizontal_acceleration_drone / F
    target_yaw = np.arctan2(target_position[1], target_position[0])
    if target_yaw > np.pi:
        target_yaw = target_yaw - 2 * np.pi
    if target_yaw < -np.pi:
        target_yaw = target_yaw + 2 * np.pi
    v_yaw = target_yaw / rotation_tau
    if abs(v_yaw) > max_ang_speed:
        v_yaw = v_yaw / abs(v_yaw) * max_ang_speed

    return ControlOutput(velocity=(cmd_linear_x, cmd_linear_y, cmd_linear_z),
                         angular_speed=_cmd_from_angular_speed(v_yaw))


class FollowControllers(object):
    """This class implements the different controllers presented in the paper ...."""

    # TODO: complete (add and document any argument so that they can be exposed as ROS params)
    def __init__(self, v1_model_path, v2_model_path, v3_model_path):
        model = FrontNet(PreActBlock, [1, 1, 1])
        ModelManager.Read(v1_model_path, model)
        #state_dict = torch.load(v1_model_path, map_location='cpu')
        rospy.loginfo(v1_model_path)
        #model.load_state_dict(state_dict['model'])
        #model.load_state_dict(torch.load(v1_model_path, map_location='cpu'))
       
        self.trainer = ModelTrainer(model)
        model2 = FrontNet(PreActBlock, [1, 1, 1])
        #smodel2.load_state_dict(torch.load(v2_model_path, map_location='cpu'))
        ModelManager.Read(v2_model_path, model2)
        self.trainer2 = ModelTrainer(model2)


        self.pose_pub = rospy.Publisher("predicted_pose", PoseStamped, queue_size=1)

        # self.v1_model = keras.models.load_model(v1_model_path)
        # self.v1_model._make_predict_function()
        # self.v1_graph = tf.get_default_graph()
        

        # self.v2_model = keras.models.load_model(v2_model_path)
        # self.v2_model._make_predict_function()
        # self.v2_graph = tf.get_default_graph()
        # self.v3_model = keras.models.load_model(v3_model_path)
        # self.v3_model._make_predict_function()
        # self.v3_graph = tf.get_default_graph()

    # This function is the only public interface that will be called by the ROS controller
    def update(self, controller, drone_velocity, frame=None, target_position=None, target_yaw=None):
        """
            Different controllers need different inputs, the rest can be left as None:
            - frame is the front camera frame as a numpy array of shape ???
              [needed by v1, v2 and v3],
            - drone_velocity is the velocity of the drone from the visual odometry
              as a list [vx, vy, vz] [needed by all],
            - target_position is the position of the target (head) as a list [x, y, z]
              [needed by exact],
            - target_yaw is the yaw [radian] of the target (i.e. the angle where the head is facing)
              as a float [needed by exact].
            Returns a ControlOutput tuple
        """
        if controller == Controller.exact:
            return self._exact_control(drone_velocity, target_position, target_yaw)
        if controller == Controller.v1:
            return self._v1_control(frame, drone_velocity)
        if controller == Controller.v2:
            return self._v2_control(frame, drone_velocity)
        # if controller == Controller.v3:
        #     return self._v3_control(frame, drone_velocity)
        raise NotImplementedError("Controller %s not implemented" % controller)

    # TODO: complete
    def _exact_control(self, drone_velocity, target_position, target_yaw):
        """
            - drone_velocity is the velocity of the drone from the visual odometry
              as a list [vx, vy, vz],
            - target_position is the position of the target (head) as a list [x, y, z],
            - target_yaw is the yaw [radian] of the target (i.e. the angle where the head is facing)
              as a float.
            Returns a ControlOutput tuple
        """
        if target_position is None or target_yaw is None:
            return ControlOutput(velocity=[0, 0, 0], angular_speed=0)
        return _controller(target_position=target_position, target_yaw=target_yaw,
                           drone_velocity=drone_velocity)

    # TODO: complete
    def _v1_control(self, frame, drone_velocity):
        """
            - frame is the front camera frame as a numpy array of shape ???,
            - drone_velocity is the velocity of the drone from the visual odometry
              as a list [vx, vy, vz],
            Returns a ControlOutput tuple
        """
        # with self.v1_graph.as_default():
        #     v1_pred = np.squeeze(self.v1_model.predict(frame))
        v1_pred = self.trainer.InferSingleSample(frame)
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "bebop"
        p = msg.pose.position
        p.x, p.y, p.z = v1_pred[:3]
        
        q = quaternion_from_euler(0, 0, v1_pred[3] + np.pi)
        msg.pose.orientation = Quaternion(*q)
        self.pose_pub.publish(msg)
        return _controller(target_position=[v1_pred[0], v1_pred[1], v1_pred[2]],
                           target_yaw=(np.pi + v1_pred[3]), drone_velocity=drone_velocity)

    # TODO: complete
    def _v2_control(self, frame, drone_velocity):
        """
            - frame is the front camera frame as a numpy array of shape ???,
            - drone_velocity is the velocity of the drone from the visual odometry
              as a list [vx, vy, vz],
            Returns a ControlOutput tuple
        """
        v1_pred = self.trainer2.InferSingleSample(frame)
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "bebop"
        p = msg.pose.position
        p.x, p.y, p.z = v1_pred[:3]
        
        q = quaternion_from_euler(0, 0, v1_pred[3] + np.pi)
        msg.pose.orientation = Quaternion(*q)
        self.pose_pub.publish(msg)
        return _controller(target_position=[v1_pred[0], v1_pred[1], v1_pred[2]],
                           target_yaw=(np.pi + v1_pred[3]), drone_velocity=drone_velocity)
    #     v_drone = np.expand_dims(np.array(drone_velocity[:2]), axis=0)
    #     with self.v2_graph.as_default():
    #         res = np.squeeze(self.v2_model.predict([frame, v_drone]))
    #     return ControlOutput(velocity=(res[0], res[1], res[2]), angular_speed=res[3])

    # def _v3_control(self, frame, drone_velocity):
    #     """
    #         - frame is the front camera frame as a numpy array of shape ???,
    #         - drone_velocity is the velocity of the drone from the visual odometry
    #           as a list [vx, vy, vz],
    #         Returns a ControlOutput tuple

    #     """
    #     with self.v1_graph.as_default():
    #         v1_pred = self.v1_model.predict(frame)
    #     v_drone = np.expand_dims(np.array(drone_velocity[:2]), axis=0)
    #     squeezed = np.squeeze(np.copy(v1_pred))
    #     v1_copy = np.expand_dims(squeezed, axis=0)
    #     with self.v3_graph.as_default():
    #         res = np.squeeze(self.v3_model.predict([v1_copy, v_drone]))
    #     return ControlOutput(velocity=tuple(res[:3]), angular_speed=res[3])
