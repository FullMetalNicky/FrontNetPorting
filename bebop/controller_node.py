#!/usr/bin/env python

from time import time

import cv2
import numpy as np

import diagnostic_msgs
import diagnostic_updater
import message_filters
import rospy
from dynamic_reconfigure.server import Server
from follow import Controller, FollowControllers
from geometry_msgs.msg import Twist, Vector3, PoseStamped, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage
from test_dario.cfg import ControllersConfig
from tf.transformations import euler_from_quaternion
from std_msgs.msg import UInt8
from sensor_msgs.msg import Joy
# from drone_arena_msgs.msg import GoToPoseAction, GoToPoseGoal
import actionlib


def frame_from_msg(msg, image_height=60, image_width=108):
    """
        Converts a jpeg image in a 3d numpy array of RGB pixels and resizes it to the given size.
      Args:
        a ros msg as a compressed BGR jpeg image.
        size: a tuple containing width and height, or None for no resizing.
      Returns:
        img: the raw, resized image as a 3d numpy array of RGB pixels.
    """
    compressed = np.fromstring(msg.data, np.uint8)
    raw = cv2.imdecode(compressed, cv2.IMREAD_COLOR)s
    img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_width, image_height))
    return img
   

def rotate_z(vector, angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.dot(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]), vector)


def twist_from_control(control):
    msg = Twist()
    msg.linear = Vector3(*control.velocity)
    msg.angular = Vector3(0, 0, control.angular_speed)
    return msg


class ROSFollowControllers(object):

    def __init__(self):
        rospy.init_node('follow_controllers')
        cmd_topic = rospy.get_param('~cmd_topic', 'cmd_vel_input')

        use_mocap = rospy.get_param("~use_mocap", True)

        self.active_controller_pub = rospy.Publisher(
            'active_controller', UInt8, latch=True, queue_size=None)

        self.controllers = FollowControllers(
            v1_model_path=rospy.get_param('~v1_model_path'),
            v2_model_path=rospy.get_param('~v2_model_path'),
            v3_model_path=rospy.get_param('~v3_model_path'))
        self.should_publish_all_controllers = rospy.get_param('~enable_debug_pubs')
        self.active_controller = Controller(rospy.get_param('~controller'))
        self.srv = Server(ControllersConfig, self.callback)
        video = message_filters.Subscriber('video', CompressedImage)
        vo_odom_drone = message_filters.Subscriber('vo_odom_drone', Odometry)
        subs = [video, vo_odom_drone]

        if use_mocap:
            mocap_odom_drone = message_filters.Subscriber('mocap_odom_drone', Odometry)
            mocap_odom_head = message_filters.Subscriber('mocap_odom_head', Odometry)
            subs += [mocap_odom_drone, mocap_odom_head]

        self.controller_pub = {controller: rospy.Publisher(
            'output_{name}'.format(name=controller.name), Twist, queue_size=1)
            for controller in Controller if controller != Controller.idle}
        self.cmd_pub = rospy.Publisher(cmd_topic, Twist, queue_size=1)

        message_filters.ApproximateTimeSynchronizer(
            subs, queue_size=10, slop=0.1).registerCallback(self.update)

        self.last_controller_duration = {}

        updater = diagnostic_updater.Updater()
        updater.setHardwareID("ML")
        updater.add("Controllers", self.controllers_diagnostics)

        def update_diagnostics(event):
            updater.update()

        rospy.Timer(rospy.Duration(1), update_diagnostics)
        freq_bounds = {'min': 4, 'max': 6}

        self.pub_freq = diagnostic_updater.HeaderlessTopicDiagnostic(
            cmd_topic, updater, diagnostic_updater.FrequencyStatusParam(freq_bounds, 0.1, 10))

        if use_mocap:
            self.start_pose = PoseStamped()
            self.start_pose.header.frame_id = 'World'
            self.start_pose.pose.position = Point(
                rospy.get_param('~start_x', 0),
                rospy.get_param('~start_y', 0),
                rospy.get_param('~start_z', 1))
            start_yaw = rospy.get_param('~start_yaw', 0)
            self.start_pose.pose.orientation.w = np.cos(start_yaw * 0.5)
            self.start_pose.pose.orientation.z = np.sin(start_yaw * 0.5)

            #self.goto = actionlib.SimpleActionClient('fence_control', GoToPoseAction)
            #self.goto.wait_for_server()
        else:
            self.goto = None
            self.start_pose = None
        # self.target_pub = rospy.Publisher('target_pose', PoseStamped, queue_size=1)

        rospy.Subscriber('joy', Joy, self.stop, queue_size=1)
        rospy.loginfo("Ready to start")

    def stop(self, msg):
        #if msg.buttons[6] == 1 and self.goto is not None and self.start_pose is not None:
        #    self.start_pose.header.stamp = rospy.Time.now()
        #    goal = GoToPoseGoal(target_pose=self.start_pose)
        #    self.goto.send_goal(goal)
        #    self.goto.wait_for_result(rospy.Duration(20))
        #    return
        old_controller = self.active_controller
        if msg.buttons[7] == 1:
            self.active_controller = Controller.idle
        elif msg.axes[5] > 0:
            self.active_controller = Controller.exact
        elif msg.axes[4] < 0:
            self.active_controller = Controller.v1
        elif msg.axes[5] < 0:
            self.active_controller = Controller.v2
        elif msg.axes[4] > 0:
            self.active_controller = Controller.v3
        else:
            return
        if old_controller != self.active_controller:
            self.srv.update_configuration(
                {'controller': self.active_controller.value,
                 'enable_debug_pubs': self.should_publish_all_controllers})
            self.active_controller_pub.publish(self.active_controller.value)

    def controllers_diagnostics(self, stat):
        stat.summary(diagnostic_msgs.msg.DiagnosticStatus.OK, "")
        for c in Controller:
            if c == Controller.idle:
                continue

            if c in self.last_controller_duration:
                value = 'last update {0:.1f} [ms]'.format(1000 * self.last_controller_duration[c])
            else:
                value = '-'
            if c == self.active_controller:
                value += ' (active)'
            stat.add(c.name, value)

    def callback(self, config, level):
        self.should_publish_all_controllers = config['enable_debug_pubs']
        old_controller = self.active_controller
        self.active_controller = Controller(config['controller'])
        if old_controller != self.active_controller:
            self.active_controller_pub.publish(self.active_controller.value)
        return config

    def update(self, video, vo_odom_drone, mocap_odom_drone=None, mocap_odom_head=None):

        active_controller = self.active_controller

        # TODO: check shape
        frame = frame_from_msg(video)
        rospy.loginfo(frame.shape)
        v = vo_odom_drone.twist.twist.linear
        drone_velocity = [v.x, v.y, v.z]

        if mocap_odom_drone is not None and mocap_odom_head is not None:
            p = mocap_odom_drone.pose.pose.position
            drone_position_world = np.array([p.x, p.y, p.z])
            p = mocap_odom_head.pose.pose.position
            head_position_world = np.array([p.x, p.y, p.z])
            q = mocap_odom_drone.pose.pose.orientation
            # We assume pitch and roll = 0 (as the camera is stabilized)
            _, _, drone_yaw_world = euler_from_quaternion([q.x, q.y, q.z, q.w])
            q = mocap_odom_head.pose.pose.orientation
            _, _, head_yaw_world = euler_from_quaternion([q.x, q.y, q.z, q.w])
            head_position = rotate_z(head_position_world - drone_position_world, -drone_yaw_world)
            head_yaw = head_yaw_world - drone_yaw_world
        else:
            head_position = None
            head_yaw = None

        last_controller_duration = {}

        if active_controller != Controller.idle:
            start = time()
            output = self.controllers.update(
                controller=active_controller, frame=frame, drone_velocity=drone_velocity,
                target_position=head_position, target_yaw=head_yaw)
            duration = time() - start
            output = twist_from_control(output)
            if self.should_publish_all_controllers:
                self.controller_pub[active_controller].publish(output)
            self.cmd_pub.publish(output)
            self.pub_freq.tick()
            last_controller_duration[active_controller] = duration

        if self.should_publish_all_controllers:
            for controller in Controller:
                if controller not in [Controller.idle, active_controller]:
                    start = time()
                    output = self.controllers.update(
                        controller=controller, frame=frame, drone_velocity=drone_velocity,
                        target_position=head_position, target_yaw=head_yaw)
                    output = twist_from_control(output)
                    duration = time() - start
                    self.controller_pub[controller].publish(output)
                    last_controller_duration[controller] = duration
        self.last_controller_duration = last_controller_duration


if __name__ == '__main__':
    ROSFollowControllers()
    rospy.spin()
