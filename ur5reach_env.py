import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pybullet as p
import time
import pybullet_data
import os
from pprint import pprint
import math
from gym import spaces
import random
from time import sleep
import transform_utils as T
import threading

class Ur5ReachEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.assets = os.path.abspath(__file__ + "/../../") + '/assets'
        # or p.DIRECT for non-graphical version
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(0, 0, -10)
        self.robot_id = 0
        self.target_id = 1
        self.table_id = 2
        self.ball_id = 3
        self.cylinder_id = 4
        self.robot_start_pos = [0, 0, 1]
        self.max_forces = [150.0, 150.0, 150.0]
        self.robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.limit_distance = 0.05 # meter
        #self.limit_distance2 = 0.05 # meter
        self.too_far = 1
        self.goal_reward = 10.0
        self.goal_reward2 = 20.0
        self.camera_yaw = p.addUserDebugParameter('Camera Yaw', 0, 180, 0)
        self.camera_pitch = p.addUserDebugParameter('Camera Pitch', -100, 0, 0)
        self.camera_distance = p.addUserDebugParameter('Camera Distance', 0, 10, 0)
        self.action_space = 41
        self.observation_space = 4
        self.reach = False  

        def step(self, action, target, a1_prec, a2_prec, a3_prec):
        #To change the angle of the vision in the simulation
        #self.user_yaw = p.readUserDebugParameter(self.camera_yaw)
        #self.user_pitch = p.readUserDebugParameter(self.camera_pitch)
        #self.user_distance = p.readUserDebugParameter(self.camera_distance)
        #p.resetDebugVisualizerCamera( cameraDistance=self.user_distance, cameraYaw=self.user_yaw, cameraPitch=self.user_pitch, cameraTargetPosition=[0,0.5,1])


        self.take_action(action, a1_prec, a2_prec, a3_prec)    
        state = self.state(action, a1_prec, a2_prec, a3_prec)
        distance = self._distance_to_target(target) 
        
        if distance > self.too_far:
            reward = -5

        elif distance > self.limit_distance:
            reward = 2*self.reward_from_distance(distance)
            
        elif distance < self.limit_distance:
            reward = self.goal_reward

        done=False
        if reward>19:
            print(reward)
            print(distance)
            done=True
            
        return state, reward, done
    
        def stepQLearning(self, action, target1, a1_prec, a2_prec, a3_prec, a4_prec):
        self.user_yaw = p.readUserDebugParameter(self.camera_yaw)
        self.user_pitch = p.readUserDebugParameter(self.camera_pitch)
        self.user_distance = p.readUserDebugParameter(self.camera_distance)
        p.resetDebugVisualizerCamera( cameraDistance=2.5, cameraYaw=210, cameraPitch=-150, cameraTargetPosition=[0,0.5,1])

        self.take_action(action, a1_prec, a2_prec, a3_prec, a4_prec)
        
        stateQLearning = self.stateQLearning(action, a1_prec, a2_prec, a3_prec, a4_prec)
        distance = self._distance_to_target(target1)
        
        if distance > self.too_far:
            reward = -1

        elif distance > self.limit_distance:
            reward = self.reward_from_distance(distance)
            print(reward)
            
        elif distance < self.limit_distance: 
            reward = self.goal_reward
            print(reward)
            print("---------------------------------------------------")

        done=False    
    
        
        if reward>9:
            print(reward)
            print(distance)
            done=True

        return stateQLearning, reward, done 
        
    def step_simu(self):
        for i in range(1, 400):
            p.stepSimulation()
    
        return True

    def _distance_to_target(self, target1):
        v1 = target1
        v2 = self.get_observation()
        distance = np.sqrt(np.sum((v1 - v2) ** 2))
        return distance
        

    def reset(self, target1):
        p.resetSimulation()
        planeId = p.loadURDF("plane.urdf")
        target_pos = target1
        self.reach = False
        self.robot_id = p.loadURDF("/home/jorand/pybullet_ws/src/gym-pybullet/gym_pybullet/envs/assets/ur_description/ur5.urdf", self.robot_start_pos,
                                  self.robot_start_orientation, flags=p.URDF_USE_SELF_COLLISION)

 
        self.target_id = p.loadURDF("/home/jorand/pybullet_ws/src/gym-pybullet/gym_pybullet/envs/assets/ur_description/ball1.urdf", target_pos)
        self.table_id = p.loadURDF("/home/jorand/pybullet_ws/src/gym-pybullet/gym_pybullet/envs/assets/ur_description/table.urdf", [0.5, 0, 0.475])
       

        for i in range(1,7):
            p.enableJointForceTorqueSensor(self.robot_id,i)
        p.stepSimulation()

        action = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40])
      
        a1_prec=random.choice([1.8, 1.56, 1.32, 1.08, 0.84, 0.6, 0.36, 0.12, -0.12, -0.36, -0.6, -0.84, -1.08, -1.32, -1.56, -1.8])
        a2_prec=random.choice([-0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.1, -1.2, -1.3])
        a3_prec=random.choice([0.8, 0.935, 1.07, 1.205, 1.34, 1.475, 1.61, 1.745, 1.88, 2.015, 2.15])
        a4_prec=random.choice([-0.5, -1, -1.5, -2])

        state = self.state()
        print ("--------------------------------------------------------")
        return state, action, a1_prec, a2_prec, a3_prec, a4_prec

    def render(self, mode='human', close=False):
        p.connect(GUI)

    def get_observation(self):
        tip_position = np.array(p.getLinkState(self.robot_id,7)[0])
        p.stepSimulation()
        #print("tip pos :")
        #print(tip_position)
        return tip_position

    def get_state(self):
        print(p.getJointState(self.robot_id, 1)[0])
        
    def terminate(self):
        p.disconnect()

    def generate_target(self):
        pi = np.pi
        phi = np.random.uniform(pi/6, pi/3)
        theta = np.random.uniform(120*pi/180, 80*pi/180)
        r = np.random.uniform(0.95, 1.2)

        z = np.cos(phi)*r
        x = np.sin(phi)*np.cos(theta)*r
        y = np.sin(phi)*np.sin(theta)*r

        #x=np.random.uniform(-0.3, 0.3)
        #y=np.random.uniform(-0.3, 0.3)
        #z=1
        target1 = (x, y, z)
        return target1
        

    def reward_from_distance(self, distance):
        return 0.8/(distance)
    
        def stateQLearning(self, action, a1_prec, a2_prec, a3_prec, a4_prec):
        s=[]
        for i in range(1):
            for j in [-1.8, -1.56, -1.32, -1.08, -0.84, -0.6, -0.36, -0.12, 0.12, 0.36, 0.6, 0.84, 1.08, 1.32, 1.56, 1.8]:
                for k in [-1.3, -1.2, -1.1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3]:
                    for l in [0.8, 0.935, 1.07, 1.205, 1.34, 1.475, 1.61, 1.745, 1.88, 2.015, 2.15]:
                        for m in [-0.5, -1, -1.5, -2]:
                            s.insert(i,(j,k,l,m))
        
        stateQLearning = s.index(self.take_action(action, a1_prec, a2_prec, a3_prec, a4_prec))
        return stateQLearning 

    def state(self):
        state = (p.getJointState(self.robot_id,1)[0], p.getJointState(self.robot_id,2)[0], p.getJointState(self.robot_id,3)[0], p.getJointState(self.robot_id,4)[0])
        return state

    def take_action(self, action, a1_prec, a2_prec, a3_prec, a4_prec):
    
        a = action

        a1=a1_prec
        a2=a2_prec
        a3=a3_prec
        a4=a4_prec

        if a == 0:
            a1=-1.8
            p.setJointMotorControl2(self.robot_id, 1, p.POSITION_CONTROL, targetPosition=a1)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 1:
            a1=-1.56
            p.setJointMotorControl2(self.robot_id, 1, p.POSITION_CONTROL, targetPosition=a1)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 2:
            a1=-1.32
            p.setJointMotorControl2(self.robot_id, 1, p.POSITION_CONTROL, targetPosition=a1)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 3:
            a1=-1.08
            p.setJointMotorControl2(self.robot_id, 1, p.POSITION_CONTROL, targetPosition=a1)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 4:
            a1=-0.84
            p.setJointMotorControl2(self.robot_id, 1, p.POSITION_CONTROL, targetPosition=a1)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 5:
            a1=-0.6
            p.setJointMotorControl2(self.robot_id, 1, p.POSITION_CONTROL, targetPosition=a1)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 6:
            a1=-0.36
            p.setJointMotorControl2(self.robot_id, 1, p.POSITION_CONTROL, targetPosition=a1)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 7:
            a1=-0.12
            p.setJointMotorControl2(self.robot_id, 1, p.POSITION_CONTROL, targetPosition=a1)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 8:
            a1=0.12
            p.setJointMotorControl2(self.robot_id, 1, p.POSITION_CONTROL, targetPosition=a1)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 9:
            a1=0.36
            p.setJointMotorControl2(self.robot_id, 1, p.POSITION_CONTROL, targetPosition=a1)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 10:
            a1=0.6
            p.setJointMotorControl2(self.robot_id, 1, p.POSITION_CONTROL, targetPosition=a1)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 11:
            a1=0.84
            p.setJointMotorControl2(self.robot_id, 1, p.POSITION_CONTROL, targetPosition=a1)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 12:
            a1=1.08
            p.setJointMotorControl2(self.robot_id, 1, p.POSITION_CONTROL, targetPosition=a1)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 13:
            a1=1.32
            p.setJointMotorControl2(self.robot_id, 1, p.POSITION_CONTROL, targetPosition=a1)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 14:
            a1=1.56
            p.setJointMotorControl2(self.robot_id, 1, p.POSITION_CONTROL, targetPosition=a1)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 15:
            a1=1.8
            p.setJointMotorControl2(self.robot_id, 1, p.POSITION_CONTROL, targetPosition=a1)
            self.step_simu()
            return ((a1, a2, a3, a4))
     #Shoulder_lift_joint
        elif a == 16:
            a2=-1.3
            p.setJointMotorControl2(self.robot_id, 2, p.POSITION_CONTROL, targetPosition=a2)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 17:
            a2=-1.2
            p.setJointMotorControl2(self.robot_id, 2, p.POSITION_CONTROL, targetPosition=a2)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 18:
            a2=-1.1
            p.setJointMotorControl2(self.robot_id, 2, p.POSITION_CONTROL, targetPosition=a2)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 19:
            a2=-0.9
            p.setJointMotorControl2(self.robot_id, 2, p.POSITION_CONTROL, targetPosition=a2)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 20:
            a2=-0.8
            p.setJointMotorControl2(self.robot_id, 2, p.POSITION_CONTROL, targetPosition=a2)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 21:
            a2=-0.7
            p.setJointMotorControl2(self.robot_id, 2, p.POSITION_CONTROL, targetPosition=a2)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 22:
            a2=-0.6
            p.setJointMotorControl2(self.robot_id, 2, p.POSITION_CONTROL, targetPosition=a2)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 23:
            a2=-0.5
            p.setJointMotorControl2(self.robot_id, 2, p.POSITION_CONTROL, targetPosition=a2)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 24:
            a2=-0.4
            p.setJointMotorControl2(self.robot_id, 2, p.POSITION_CONTROL, targetPosition=a2)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 25:
            a2=-0.3
            p.setJointMotorControl2(self.robot_id, 2, p.POSITION_CONTROL, targetPosition=a2)
            self.step_simu()
            return ((a1, a2, a3, a4))
        #Elbow_joint
        elif a == 26:
            a3=2.15
            p.setJointMotorControl2(self.robot_id, 3, p.POSITION_CONTROL, targetPosition=a3)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 27:
            a3=2.015
            p.setJointMotorControl2(self.robot_id, 3, p.POSITION_CONTROL, targetPosition=a3)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 28:
            a3=1.88
            p.setJointMotorControl2(self.robot_id, 3, p.POSITION_CONTROL, targetPosition=a3)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 29:
            a3=1.745
            p.setJointMotorControl2(self.robot_id, 3, p.POSITION_CONTROL, targetPosition=a3)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 30:
            a3=1.61
            p.setJointMotorControl2(self.robot_id, 3, p.POSITION_CONTROL, targetPosition=a3)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 31:
            a3=1.475
            p.setJointMotorControl2(self.robot_id, 3, p.POSITION_CONTROL, targetPosition=a3)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 32:
            a3=1.34
            p.setJointMotorControl2(self.robot_id, 3, p.POSITION_CONTROL, targetPosition=a3)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 33:
            a3=1.205
            p.setJointMotorControl2(self.robot_id, 3, p.POSITION_CONTROL, targetPosition=a3)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 34:
            a3=1.07
            p.setJointMotorControl2(self.robot_id, 3, p.POSITION_CONTROL, targetPosition=a3)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 35:
            a3=0.935
            p.setJointMotorControl2(self.robot_id, 3, p.POSITION_CONTROL, targetPosition=a3)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 36:
            a3=0.8
            p.setJointMotorControl2(self.robot_id, 3, p.POSITION_CONTROL, targetPosition=a3)
            self.step_simu()
            return ((a1, a2, a3, a4))
        #Wrist 1 Joint
        elif a == 37:
            a3=-0.5
            p.setJointMotorControl2(self.robot_id, 4, p.POSITION_CONTROL, targetPosition=a4)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 38:
            a3=-1
            p.setJointMotorControl2(self.robot_id, 4, p.POSITION_CONTROL, targetPosition=a4)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 39:
            a3=-1.5
            p.setJointMotorControl2(self.robot_id, 4, p.POSITION_CONTROL, targetPosition=a4)
            self.step_simu()
            return ((a1, a2, a3, a4))
        elif a == 40:
            a3=-2
            p.setJointMotorControl2(self.robot_id, 4, p.POSITION_CONTROL, targetPosition=a4)
            self.step_simu()
            return ((a1, a2, a3, a4))



    def action_sample(self, a1_prec, a2_prec, a3_prec, a4_prec):
        #Choose randomly the action to take
        
        a = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40])
        
        # The previous action taken
        a1_p = a1_prec
        a2_p = a2_prec
        a3_p = a3_prec
        a4_p = a4_prec

        if a == 0:
            a1_p=-1.8
            a1_p=a1_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
            #return (p.setJointMotorControl2(self.robot_id, 1, p.POSITION_CONTROL, targetPosition=a1), (a1, a2, a3), a)
        elif a == 1:
            a1_p=-1.56
            a1_p=a1_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
            #return (p.setJointMotorControl2(self.robot_id, 1, p.POSITION_CONTROL, targetPosition=a1), (a1, a2, a3), a)
        elif a == 2:
            a1_p=-1.32
            a1_p=a1_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
            #return (p.setJointMotorControl2(self.robot_id, 1, p.POSITION_CONTROL, targetPosition=a1), (a1, a2, a3), a)
        elif a == 3:
            a1_p=-1.08
            a1_p=a1_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
            #return (p.setJointMotorControl2(self.robot_id, 1, p.POSITION_CONTROL, targetPosition=a1), (a1, a2, a3), a)
        elif a == 4:
            a1_p=-0.84
            a1_p=a1_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
            #return (p.setJointMotorControl2(self.robot_id, 1, p.POSITION_CONTROL, targetPosition=a1), (a1, a2, a3), a)
        elif a == 5:
            a1_p=-0.6
            a1_p=a1_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
            #return (p.setJointMotorControl2(self.robot_id, 1, p.POSITION_CONTROL, targetPosition=a1), (a1, a2, a3), a)
        elif a == 6:
            a1_p=-0.36
            a1_p=a1_prec
            return a, a1_prec, a2_prec, a3_prec,a4_prec
            #return (p.setJointMotorControl2(self.robot_id, 1, p.POSITION_CONTROL, targetPosition=a1), (a1, a2, a3), a)
        elif a == 7:
            a1_p=-0.12
            a1_p=a1_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
            #return (p.setJointMotorControl2(self.robot_id, 1, p.POSITION_CONTROL, targetPosition=a1), (a1, a2, a3), a)
        elif a == 8:
            a1_p=0.12
            a1_p=a1_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
            #return (p.setJointMotorControl2(self.robot_id, 1, p.POSITION_CONTROL, targetPosition=a1), (a1, a2, a3), a)
        elif a == 9:
            a1_p=0.36
            a1_p=a1_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
            #return (p.setJointMotorControl2(self.robot_id, 1, p.POSITION_CONTROL, targetPosition=a1), (a1, a2, a3), a)
        elif a == 10:
            a1_p=0.6
            a1_p=a1_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
        elif a == 11:
            a1_p=0.84
            a1_p=a1_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
        elif a == 12:
            a1_p=1.08
            a1_p=a1_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
        elif a == 13:
            a1_p=1.32
            a1_p=a1_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
        elif a == 14:
            a1_p=1.56
            a1_p=a1_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
        elif a == 15:
            a1_p=1.8
            a1_p=a1_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
     #Shoulder_lift_joint
        elif a == 16:
            a2_p=-1.3
            a2_p=a2_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
            #return (p.setJointMotorControl2(self.robot_id, 2, p.POSITION_CONTROL, targetPosition=a2), (a1, a2, a3), a)
        elif a == 17:
            a2_p=-1.2
            a2_p=a2_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
            #return (p.setJointMotorControl2(self.robot_id, 2, p.POSITION_CONTROL, targetPosition=a2), (a1, a2, a3), a)
        elif a == 18:
            a2_p=-1.1
            a2_p=a2_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
            #return (p.setJointMotorControl2(self.robot_id, 2, p.POSITION_CONTROL, targetPosition=a2), (a1, a2, a3), a)
        elif a == 19:
            a2_p=-0.9
            a2_p=a2_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
            #return (p.setJointMotorControl2(self.robot_id, 1, p.POSITION_CONTROL, targetPosition=a2), (a1, a2, a3), a)
        elif a == 20:
            a2_p=-0.8
            a2_p=a2_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
            #return (p.setJointMotorControl2(self.robot_id, 1, p.POSITION_CONTROL, targetPosition=a2), (a1, a2, a3), a)
        elif a == 21:
            a2_p=-0.7
            a2_p=a2_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
        elif a == 2:
            a2_p=-0.6
            a2_p=a2_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
        elif a == 23:
            a2_p=-0.5
            a2_p=a2_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
        elif a == 24:
            a2_p=-0.4
            a2_p=a2_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
        elif a == 25:
            a2_p=-0.3
            a2_p=a2_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
        #Elbow_joint
        elif a == 26:
            a3_p=2.15
            a3_p=a3_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
            #return (p.setJointMotorControl2(self.robot_id, 3, p.POSITION_CONTROL, targetPosition=a3), (a1, a2, a3), a)
        elif a == 27:
            a3_p=2.015
            a3_p=a3_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
            #return (p.setJointMotorControl2(self.robot_id, 3, p.POSITION_CONTROL, targetPosition=a3), (a1, a2, a3), a)
        elif a == 28:
            a3_p=1.88
            a3_p=a3_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
            #return (p.setJointMotorControl2(self.robot_id, 3, p.POSITION_CONTROL, targetPosition=a3), (a1, a2, a3), a)
        elif a == 29:
            a3_p=1.745
            a3_p=a3_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
            #return (p.setJointMotorControl2(self.robot_id, 1, p.POSITION_CONTROL, targetPosition=a3), (a1, a2, a3), a)
        elif a == 30:
            a3_p=1.61
            a3_p=a3_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
            #return (p.setJointMotorControl2(self.robot_id, 1, p.POSITION_CONTROL, targetPosition=a3), (a1, a2, a3), a)
        elif a == 31:
            a3_p=1.475
            a3_p=a3_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
        elif a == 32:
            a3_p=1.34
            a3_p=a3_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
        elif a == 33:
            a3_p=1.205
            a3_p=a3_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
        elif a == 34:
            a3_p=1.07
            a3_p=a3_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
        elif a == 35:
            a3_p=0.935
            a3_p=a3_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
        elif a == 36:
            a3_p=0.8
            a3_p=a3_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
        #Writs 1 Joint
        elif a == 37:
            a4_p=-0.5
            a4_p=a4_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
        elif a == 38:
            a4_p=-1
            a4_p=a4_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
        elif a == 39:
            a4_p=-1.5
            a4_p=a4_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec
        elif a == 40:
            a4_p=-2
            a4_p=a4_prec
            return a, a1_prec, a2_prec, a3_prec, a4_prec

        
if __name__ == ('__main__'):
    env = Ur5ReachEnv()
    env.reset()
    env.step([1,1,1,1,1,1])


