from sheeprl.envs.controller import Supervisor
import math
import random
import torch
from transforms3d import quaternions

from abc import ABC
import numpy as np
import cv2
from numpy import inf

from gymnasium import spaces
import gymnasium as gym

import pickle

# from opendr.perception.object_detection_2d import YOLOv5DetectorLearner
# from opendr.engine.data import Image
#from opendr.perception.object_detection_2d.utils.class_filter_wrapper import FilteredLearnerWrapper

from sheeprl.envs.data import Image
#from yolo_v5 import YOLOv5DetectorLearner

class Env(gym.Env, ABC):
    metadata = {'render.modes': ['human']}
    _supervisor_instance = None

    def __init__(self, seed=1, action_space='continuous', observation_space=['rgb', 'lidar', 'world_info'], kb_control=False):
        super(Env, self).__init__()

        # Webots Environment
        # Seed
        self.seed = seed
        # continuous or discrete actionspace:
        self.a_space = action_space
        self.o_space = observation_space
        # Set seeds
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Webots supervisor, robot and devices
        #self.supervisor = Supervisor()

        if Env._supervisor_instance is None:
            Env._supervisor_instance = Supervisor()
            self.supervisor = Env._supervisor_instance
        else:
            self.supervisor = Env._supervisor_instance

        self.robot = self.supervisor.getSelf()

        print("DEVICES")
        print(self.supervisor.devices)
        # Motors
        self.left_motor = self.supervisor.getDevice("left wheel motor")
        self.right_motor = self.supervisor.getDevice("right wheel motor")
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        # Robot rotation and position
        self.rotation = self.robot.getField('rotation')
        self.position = self.robot.getField('translation')
        self.timestep = int(self.supervisor.getBasicTimeStep())
        # Root and children of world structure
        self.root = self.supervisor.getRoot()
        self.children = self.root.getField('children')

        #print(self.supervisor.getFromDef("F"))
        #print(self.supervisor.get)
        info = self.supervisor.getFromDef("worldinfo")
        self.gravity = info.getField('gravity')
        print("Gravity")
        print(self.gravity)

        # Robot camera compass, gyro, lidar and touchsensor
        self.camera = self.supervisor.getDevice('camera')
        self.camera.enable(self.timestep)
        self.compass = self.supervisor.getDevice('compass')
        self.compass.enable(self.timestep)
        self.gyro = self.supervisor.getDevice('gyro')
        self.gyro.enable(self.timestep)
        self.lidar = self.supervisor.getDevice('LDS-01')
        self.lidar.enable(self.timestep)
        self.sensor = self.supervisor.getDevice('touch sensor')
        self.sensor.enable(self.timestep)
        self.lidar_main_motor = self.supervisor.getDevice('LDS-01_main_motor')
        self.lidar_secondary_motor = self.supervisor.getDevice('LDS-01_secondary_motor')
        self.lidar_main_motor.setPosition(float('inf'))
        self.lidar_secondary_motor.setPosition(float('inf'))
        self.lidar_main_motor.setVelocity(30.0)
        self.lidar_secondary_motor.setVelocity(60.0)

        if kb_control: # enable keyboard control
            self.supervisor.keyboard.enable(self.timestep)
        # ----------------------------- GYM STUFF ----------------------------- #
        # Action Spaces
        if action_space == 'discrete':
            self.action_space = spaces.Discrete(6)
            self.gravity.setSFFloat(0.0)
        if action_space == 'continuous':
            self.action_space = spaces.Box(-1.0, 1.0, (3,))
        # Reward Range
        # self.reward_range = [-10, 10]
        # Observation Spaces
        self.padding = 15  # padding for lidar data
        self.padding1 = 60
        self.vector_size = self.lidar.getHorizontalResolution() + 2 * self.padding
        self.img_size = (270, 480, 3)
        self.framestack = 1

        self.episode_readings = []

        if observation_space == ['rgb']:
            self.observation_space = spaces.Box(low=0, high=255, shape=self.img_size, dtype=np.uint8)
        else:
            observation_dict = {}
            if 'rgb' in observation_space:
                observation_dict['rgb'] = spaces.Box(0, 255, self.img_size, dtype=np.uint8)
            if 'lidar' in observation_space:
                observation_dict['lidar'] = spaces.Box(0, 1, (self.framestack * self.vector_size,), dtype=np.float64)
            if 'world_info' in observation_space:
                observation_dict['world_info'] = spaces.Box(0, 1, (self.framestack * 5,), dtype=np.float64)

            self.observation_space = spaces.Dict(
                spaces=observation_dict
            )
        self.lidar_obs = []
        self.world_obs = []
        self.obs = {}
        self.reward_range = [-10, 10]
        self.metadata = None
        self.step_counter = 0
        self.frameskip = 4
        self.info = None
        # for reward
        self.distance, self.prev_distance, self.angle, self.prev_angle, self.angle_pi =\
            100.0, 100.0, 100.0, 100.0, 100.0
        self.target_distance = 1.5
        self.just_reset = False
        self.min_distance = 100.0
        self.min_angle = 10.0
        self.motor_speeds = [0, 0]
        self.keyboard_control = kb_control
        self.distance_to_target = 100
        self.cumulative_reward = 0
        # Yolov5
        self.person_detector, self.filtered = None, None
        self.load_models()
        self.found = False
        self.closest_robot_entrance, self.closest_human_entrance = None, None
        self.next_target = None # 1st target is closest_robot_entrance, 2nd is closest_human_entrance, 3rd is human
        self.angle_to_target = 10
        self.angle_pi_to_target = 10
        self.target = None
        # rooms [x,y, entrance]:
        self.rooms = [[(-9, -6), (-9, -6), (-5, -7.5)],
                 [(1, 9), (-9, -6), (0.66, -5)],
                 [(6, 9), (6, 9), (6.6, 5)],
                 [(1, 4), (6, 9), (0.5, 5)],
                 [(-9, -6), (6, 9), (-5, 7.5)],
                 [(-9, -6), (1, 4), (-5, 0.6)]]


    def load_models(self):
        self.person_detector = None #YOLOv5DetectorLearner(model_name='yolov5s', device='cuda')
        self.filtered = None #FilteredLearnerWrapper(self.person_detector, allowed_classes=['person'])

    def reset(self, seed=None):
        self.number_of_steps_not_moving = 0
        # self.episode_readings = []
        self.moving_human_direction = 1
        self.angle_to_target = 10
        self.angle_pi_to_target = 10
        self.just_reset = True
        self.closest_robot_entrance, self.closest_human_entrance = None, None
        self.next_target = None
        self.min_distance = 100.0
        self.min_angle = 10.0
        self.found = False
        (self.distance, self.prev_distance, self.angle, self.prev_angle,
         self.distance_to_target, self.angle_pi) = 100.0, 100.0, 100.0, 100.0, 100.0, 100.0
        self.supervisor.simulationReset()
        self.supervisor.simulationResetPhysics()
        self.supervisor.step(self.timestep)
        self.replace_humans()
        self.motor_speeds = [0, 0]
        room_x = self.get_random_room()
        room_y = self.get_random_room()
        x = random.uniform(room_x[0], room_x[1])
        y = random.uniform(room_y[0],room_y[1])
        self.position.setSFVec3f([x, y, 0.0])
        for room in self.rooms:
            if room[0][0] < x < room[0][1] and room[1][0] < y < room[1][1]:
                self.closest_robot_entrance = room[2]

        self.define_target()

        for i in range(random.randint(3, 40)):
            self.left_motor.setPosition(float('inf'))  # NOQA
            self.right_motor.setPosition(float('inf'))  # NOQA
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)
            # Step world forward
            self.supervisor.step(self.timestep)
            self.get_reward()

        if self.sensor.getValue() != 0.0 or self.distance < 0.2:
            self.reset()

        self.cumulative_reward = 0

        # Grab new observation
        obs = self.get_obs()
        self.get_reward()
        self.step_counter = 0
        return obs, self.info


    def step(self, action):
        info = {"TimeLimit.truncated": False}
        self.step_counter += 1
        done = False
        truncated = False
        collided = False
        reward = 0
        self.found = False
        
        if self.step_counter % 25 == 0:
            self.moving_human_direction *= -1 

        robot_pos = np.array(self.position.getSFVec3f())  # robot position
        
        distances_to_moving_humans = []
        for moving_human, moving_human_position in zip(self.moving_humans, self.moving_humans_positions):
            ori = np.array(moving_human.getOrientation()).reshape([3, 3])
            b = np.array([self.moving_human_direction * 0.15, 0, 0])
            new_p = ori.dot(b) + moving_human_position.getSFVec3f()
            new_p[2] = 0.0004
            moving_human_position.setSFVec3f(list(new_p))

            moving_human_pos = np.array(moving_human_position.getSFVec3f())
            target_vector = np.array(robot_pos - moving_human_pos)
            distance_to_target = np.linalg.norm(target_vector)
            distances_to_moving_humans.append(distance_to_target)

        if self.keyboard_control:
            key = self.supervisor.keyboard.getKey()
            action = self.apply_key(key)

        for i in range(self.frameskip):
            self.apply_action(action)
            self.supervisor.step(self.timestep)
            reward += self.get_reward()

        # penalize backwards movement
        if self.a_space == 'continuous':
            if self.motor_speeds[0] < 0.0 and self.motor_speeds[1] < 0.0 and reward > 0:
                reward *= -1

        if self.a_space == 'discrete':
            if action == 1 and reward > 0:
                reward *= -1

        # Grab new observation
        obs = self.get_obs()
        
        #print(self.step_counter)
        if self.step_counter == 3500:
            reward += 0
            done = True
            info = {"TimeLimit.truncated": True}
            truncated = True

        if self.sensor.getValue() != 0.0 or self.distance_to_target < 0.3:
            done = True
            reward = -10
            collided = True
        
        for distance_to_mov_human in distances_to_moving_humans:
            if distance_to_mov_human < 0.3:
                done = True
                reward = -10
                collided = True

        if self.a_space == 'discrete' and action == 4 and self.distance_to_target > self.target_distance + 1.0:
            reward -= 0.0001 * self.frameskip # penalize DL deployment outside of area
        if self.a_space == 'continuous' and action[2] > 0.0 and self.distance_to_target > self.target_distance + 1.0:
            reward -= (0.0001 * self.frameskip)  # penalize DL deployment outside of area

        if 0.3 < self.distance_to_target < self.target_distance + 1.0 and self.angle_pi < 0.1:
            done = True
            reward += 100
        # if reward != 0.0:
        #     print('Step: ', reward)
        self.cumulative_reward += round(reward, 4)
        # print(self.motor_speeds)
        # print('Step reward: ', reward)
        if done:
            print('Episode reward is {} Steps: {}'.format(self.cumulative_reward, self.step_counter))
            print('Truncated and collision: ', truncated, collided)

        print(self.step_counter, reward)
        return obs, reward, done, truncated, info

    def apply_action(self, action):
        if self.a_space == 'discrete':
            self.motor_speeds = [0.0, 0.0]
            if action == 0:  # Move forward
                ori = np.array(self.robot.getOrientation()).reshape([3, 3])
                b = np.array([0.05, 0, 0])
                new_p = ori.dot(b) + self.position.getSFVec3f()
                new_p[2] = 0.0004
                self.position.setSFVec3f(list(new_p))
            elif action == 1:  # Move Backwards
                ori = np.array(self.robot.getOrientation()).reshape([3, 3])
                b = np.array([-0.05, 0, 0])
                new_p = ori.dot(b) + self.position.getSFVec3f()
                new_p[2] = 0.0004
                self.position.setSFVec3f(list(new_p))
            elif action == 2:  # turn left
                rotation = self.rotation.getSFRotation()
                q1 = quaternions.axangle2quat(rotation[0:3], rotation[3])
                q2 = quaternions.axangle2quat([0, 0, 1], 3 * 3.14 / 180)
                q = quaternions.qmult(q1, q2)
                vec, angle = quaternions.quat2axangle(q)
                new_rotation = [vec[0], vec[1], vec[2], angle]
                self.rotation.setSFRotation(new_rotation)
            elif action == 3:  # turn right
                rotation = self.rotation.getSFRotation()
                q1 = quaternions.axangle2quat(rotation[0:3], rotation[3])
                q2 = quaternions.axangle2quat([0, 0, 1], -3 * 3.14 / 180)
                q = quaternions.qmult(q1, q2)
                vec, angle = quaternions.quat2axangle(q)
                new_rotation = [vec[0], vec[1], vec[2], angle]
                self.rotation.setSFRotation(new_rotation)
            elif action == 4: # apply person detection
                pass
            elif action == 5:  # Noop
                pass

        if self.a_space == 'continuous':
            self.motor_speeds[0] = action[0]
            self.motor_speeds[1] = action[1]
            self.set_velocity(self.motor_speeds[0], self.motor_speeds[1])
            if action[2] > 0.0:
                self.get_dl_obs()

    def apply_key(self, key):
        if key == 315:  # forward
            if self.a_space == 'discrete':
                action = 0
            else:
                action = [1.0, 1.0, 1.0]
        elif key == 317:  # backward
            if self.a_space == 'discrete':
                action = 1
            else:
                action = [-1.0, -1.0, 1.0]
        elif key == 314:  # left
            if self.a_space == 'discrete':
                action = 2
            else:
                action = [-1.0, 1.0, 1.0]
        elif key == 316:  # right
            if self.a_space == 'discrete':
                action = 3
            else:
                action = [1.0, -1.0, 1.0]
        elif key == 32:
            if self.a_space == 'discrete':
                action = 4
            else:
                action = [0.0, 0.0, 1.0]
        else:
            if self.a_space == 'discrete':
                action = 5
            else:
                action = [0.0, 0.0, 1.0]
        return action

    def get_dl_obs(self):
        cameraData = self.camera.getImage()
        if cameraData:
            frame = np.frombuffer(cameraData, np.uint8).reshape(
                (self.camera.getHeight(), self.camera.getWidth(), 4))
            frame = frame[:, :, :3]
        self.deploy_dl(frame)

    def get_obs(self):
        if 'rgb' in self.o_space:
            cameraData = self.camera.getImage()
            if cameraData:
                frame = np.frombuffer(cameraData, np.uint8).reshape(
                    (self.camera.getHeight(), self.camera.getWidth(), 4))
                frame = frame[:, :, :3]
                self.obs['rgb'] = frame

        if 'lidar' in self.o_space:
            # Lidar data is an array, 0 is behind the robot incrementing clockwise. Front of robot is Lidarpoints/2
            lidarData = self.lidar.getRangeImage()
            lidarData = np.asarray(lidarData)
            lidarData[lidarData == inf] = 12.5
            lidarData = lidarData / 12.5
            lidarData = 1 - lidarData
            # Apply Circular Padding
            lidarData_left = lidarData[:self.padding]
            lidarData_right = lidarData[-self.padding:]
            lidarData = np.concatenate((lidarData, lidarData_left), axis=0)
            lidarData = np.concatenate((lidarData_right, lidarData ), axis=0)
            if self.just_reset:
                self.lidar_obs = []
                for i in range(self.framestack):
                    self.lidar_obs.append(lidarData)
            else:
                self.lidar_obs = self.lidar_obs[1:]
                self.lidar_obs.append(lidarData)
            lidar_obs = np.asarray(self.lidar_obs)
            lidar_obs = lidar_obs.flatten()
            self.obs['lidar'] = lidar_obs

        if 'world_info' in self.o_space:
            worldData = [1.0 - round(self.distance_to_target / 29, 4),
                         round((self.angle_to_target + np.pi) / (2 * np.pi),  4),
                         round(1 - (self.angle_pi_to_target / np.pi), 4),
                         round((self.motor_speeds[1] + 1) / 2, 4),
                         round((self.motor_speeds[0] + 1) / 2, 4)]
            if self.just_reset:
                self.world_obs = []
                for i in range(self.framestack):
                    self.world_obs.append(worldData)
            else:
                self.world_obs = self.world_obs[1:]
                self.world_obs.append(worldData)
            world_obs = np.asarray(self.world_obs)
            world_obs = world_obs.flatten()
            self.obs['world_info'] = world_obs
        self.just_reset = False

        if self.o_space == ['rgb']:
            return frame
        else:
            return self.obs

    def get_reward(self):
        reward = 0

        robot_pos = np.array(self.position.getSFVec3f())  # robot position
        human_pos = np.array(self.human_position.getSFVec3f())
        # distance to final target - human
        target_vector = np.array(robot_pos - human_pos)
        self.distance_to_target = np.linalg.norm(target_vector)
        #print(self.distance_to_target)

        # distance to next target
        robot_next_target = np.array(robot_pos - self.next_target)
        distance = np.linalg.norm(robot_next_target)
        robot_ntarget_vector = robot_next_target / distance  # unit vector from drone to next target

        # drone orientation
        ori_drone = np.array(self.robot.getOrientation()).reshape([3, 3])

        # angles between vector drone-next_target and drone-forward_direction_of_drone
        b = np.array([1, 0, 0])
        new_p = ori_drone.dot(b) + robot_pos
        drone_forward = robot_pos - new_p  # unit vector drone-forward_direction

        dot_ntarget_drone = drone_forward.dot(robot_ntarget_vector)
        cross_ntarget_drone = np.cross(drone_forward, robot_ntarget_vector)
        angle = np.arctan2(cross_ntarget_drone, dot_ntarget_drone)[2]
        angle_pi = np.arccos(dot_ntarget_drone)

        # angles between vector drone-final_target and drone-forward_direction_of_drone
        robot_target_vector = target_vector / self.distance_to_target
        dot_target_drone = drone_forward.dot(robot_target_vector)
        cross_target_drone = np.cross(drone_forward, robot_target_vector)
        self.angle_to_target = np.arctan2(cross_target_drone, dot_target_drone)[2]
        self.angle_pi_to_target = np.arccos(dot_target_drone)
        # compute individual rewards
        reward += self.get_d_reward(distance)  # + self.get_a_reward(angle)
        reward += self.get_a_reward(angle_pi)
        # reward -= self.get_efficient_reward(angle, distance)

        # define next target if we reached previous
        if distance <= 0.30:
            if self.closest_human_entrance is not None and self.closest_robot_entrance is None:
                self.closest_human_entrance = None
            if self.closest_robot_entrance is not None:
                self.closest_robot_entrance = None
            # self.define_target()

        self.angle_pi = angle_pi
        self.distance = distance
        self.angle = angle
        self.prev_angle = angle_pi
        self.prev_distance = distance
        return reward

    def get_a_reward(self, angle):
        if self.a_space == 'discrete':
            a_reward = 6 * (self.prev_angle - angle)
        if self.a_space == 'continuous':
            a_reward = 2 * round((abs(round(self.prev_angle, 1)) - abs(round(angle, 1))), 2)

        # if angle < self.min_angle:
        #     a_reward = 2 * round(self.min_angle - angle, 4)
        #     self.min_angle = angle
        # else:
        #     a_reward = 0
        # if self.distance_to_target < 1.0:
        #     a_reward = 0
        self.prev_angle = angle
        # if a_reward != 0:
        #     print('Angle :', round(a_reward / 10, 2))
        return round(a_reward / 10, 2)

    def get_d_reward(self, distance):
        if self.a_space == 'discrete':
            d_reward = self.prev_distance - distance

            # if abs((self.prev_distance - distance)) < 0.001:
            #     self.number_of_steps_not_moving += 1
            
            # if self.number_of_steps_not_moving > 25:
            #     d_reward -= 1
            #     self.number_of_steps_not_moving = 0
            
            # print("Distance reward")
            # print(d_reward) 


        if self.a_space == 'continuous':
            d_reward = 5 * (round(self.prev_distance, 4) - round(distance, 4))
        # if distance < self.min_distance:
        #     d_reward = 5 * round(self.min_distance - distance, 4)
        #     self.min_distance = distance
        # else:
        #     d_reward = 0
        # if self.distance_to_target < 1.5 or d_reward < 0.01:
        #     d_reward = 0
        self.prev_distance = distance
        # if d_reward != 0:
        #     print('Distance,: ', round(2 * d_reward / 10, 3))
        return round(2 * d_reward / 10, 3)

    def get_efficient_reward(self, angle, distance):
        turn_penalty = 0.1 * (abs(angle - self.prev_angle))
        # distance_penalty = 0.1 * (abs(distance - self.prev_distance))
        efficency_reward = turn_penalty # + distance_penalty
        return efficency_reward

    def define_target(self):
        # Define closest target
        # if self.closest_robot_entrance == self.closest_human_entrance:
        #     self.closest_robot_entrance = None
        #     self.closest_human_entrance = None
        # if self.closest_robot_entrance is not None:
        #     self.next_target = np.array([self.closest_robot_entrance[0], self.closest_robot_entrance[1], 0.0])
        # if self.closest_human_entrance is not None and self.closest_robot_entrance is None:
        #     self.next_target = np.array([self.closest_human_entrance[0], self.closest_human_entrance[1], 0.0])
        # if self.closest_human_entrance is None and self.closest_robot_entrance is None:
        #     self.next_target = np.array(self.human_position.getSFVec3f())
        #     self.next_target[2] = 0.0
        self.next_target = np.array(self.human_position.getSFVec3f())
        self.next_target[2] = 0.0
        self.target = self.next_target
        # target_pos = self.target.getField('translation')
        # # print(self.next_target)
        # target_pos.setSFVec3f([self.next_target[0], self.next_target[1], self.next_target[2]])
        self.supervisor.step(self.timestep)


    def set_velocity(self, v_left, v_right):
        """
        Sets the two motor velocities.
        :param v_left: velocity value for left motor
        :type v_left: float
        :param v_right: velocity value for right motor
        :type v_right: float
        """
        self.left_motor.setPosition(float('inf'))  # NOQA
        self.right_motor.setPosition(float('inf'))  # NOQA
        self.left_motor.setVelocity(v_left * 6.0)  # NOQA
        self.right_motor.setVelocity(v_right * 6.0)  # NOQA

    def filter_persons(self, frame):
        filtered_results = self.filtered.infer(frame)
        return filtered_results

    def bbox_to_np(self, bbox):
        bbox_np = np.asarray(
            [bbox.left, bbox.top, bbox.left + bbox.width, bbox.top + bbox.height, bbox.confidence, bbox.name],
            dtype=object)
        return bbox_np

    def deploy_dl(self, frame):
        img = Image(frame)
        results = self.filter_persons(img)
        if results:
            bounding_boxes = np.asarray([self.bbox_to_np(bbox) for bbox in results.data])
            boxes = bounding_boxes[:, :4]
            scores = bounding_boxes[:, 4]
            for idx, box in enumerate(boxes):
                if scores[idx] > 0.5:
                    self.found = True
                    # (startX, startY, endX, endY) = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        # print('Run yolo, found? :', self.found)

    def set_eval(self):
        self.eval = True

    def unset_eval(self):
        self.eval = False

    def replace_humans(self):
        rnd = random.randint(1, 20)
        self.supervisor.step(self.timestep)
        if rnd < 10:
            human_model = 'human_0' + str(rnd) + '_standing'
            self.children.importMFNodeFromString(2, 'DEF ' + human_model + ' ' + human_model + ' {}')
            self.human = self.supervisor.getFromDef(human_model)
        else:
            human_model = 'human_' + str(rnd) + '_standing'
            self.children.importMFNodeFromString(2, 'DEF ' + human_model + ' ' + human_model + ' {}')
            self.human = self.supervisor.getFromDef(human_model)
        self.supervisor.step(self.timestep)
        room_x = self.get_random_room()
        room_y = self.get_random_room()
        x = random.uniform(room_x[0], room_x[1])
        y = random.uniform(room_y[0],room_y[1])

        self.human_position = self.human.getField('translation')
        self.human_position.setSFVec3f([x, y, 0])
        robot_pos = np.array(self.position.getSFVec3f())
        for room in self.rooms:
            if room[0][0] < x < room[0][1] and room[1][0] < y < room[1][1]:
                self.closest_human_entrance = room[2]

        rotation = self.human.getField('rotation')
        rotation.setSFRotation([0, 0, 1, random.randint(0, 50) * 0.1309])

        #### TEST ############

        self.moving_humans = []
        self.moving_humans_positions = []

        for i in range(1,0):
            model_index = i
            if i == rnd:
                model_index = 8
            self.supervisor.step(self.timestep)
            # if rnd < 10:
            human_model = 'human_0' + str(model_index) + '_standing'
            self.children.importMFNodeFromString(2, 'DEF ' + human_model + ' ' + human_model + ' {}')
            moving_human = self.supervisor.getFromDef(human_model)
            # else:
            #     human_model = 'human_' + str(i) + '_standing'
            #     self.children.importMFNodeFromString(2, 'DEF ' + human_model + ' ' + human_model + ' {}')
            #     moving_human = self.supervisor.getFromDef(human_model)
            self.supervisor.step(self.timestep)
            room_x = self.get_random_room()
            room_y = self.get_random_room()
            x = random.uniform(room_x[0], room_x[1])
            y = random.uniform(room_y[0],room_y[1])

            moving_human_position = moving_human.getField('translation')
            moving_human_position.setSFVec3f([x, y, 0])
            robot_pos = np.array(self.position.getSFVec3f())
            for room in self.rooms:
                if room[0][0] < x < room[0][1] and room[1][0] < y < room[1][1]:
                    closest_moving_human_entrance = room[2]

            rotation = moving_human.getField('rotation')
            rotation.setSFRotation([0, 0, 1, random.randint(0, 50) * 0.1309])
            self.moving_humans.append(moving_human)
            self.moving_humans_positions.append(moving_human_position)

    def get_random_room(self):
        return random.choice([[-9, -6], [-4, -1], [1, 4], [6, 9]])
