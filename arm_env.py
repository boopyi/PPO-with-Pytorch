import pybullet as p
import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np







class Env(gym.Env):
    def __init__(self,render_mode="none"):
        super().__init__()


        p.connect(p.GUI if render_mode=="human" else p.DIRECT)
        self.action_space = spaces.Box(low=-3.14, high=3.14, shape=(4,), dtype=np.float32)
        urdf_path = r"C:\Users\User\Desktop\roarm_description\urdf\roarm_description.urdf"
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
        self.robot_id=p.loadURDF(urdf_path,useFixedBase=True)
        self.point_vis=p.createVisualShape(p.GEOM_SPHERE,0.05,rgbaColor=[1, 0, 0, 1])
        self.point=p.createMultiBody(0,self.point_vis,basePosition=[0,0,0])
    def get_obs(self):
        obs=np.zeros((11,),dtype=np.float32)
        for i in range(p.getNumJoints(self.robot_id)-2):
            obs[i]=p.getJointState(self.robot_id,i)[0]
            obs[i+4]=p.getJointState(self.robot_id,i)[1]
        obs[8]=self.target_pos[0]
        obs[9]=self.target_pos[1]
        obs[10]=self.target_pos[2]
        return obs
        
    def reset(self,seed=None,options=None):
        self.steps_since_reset=0
        
        for i in range(p.getNumJoints(self.robot_id)-2):
            p.resetJointState(self.robot_id,i,0)
        self.target_pos=np.random.uniform(low=[0.1, -0.2, 0.05],high=[0.4,0.2,0.3])
        p.resetBasePositionAndOrientation(self.point,self.target_pos,[0,0,0,1])
            
        return (self.get_obs(),{})
    def step(self,action):
        
        self.steps_since_reset+=1
        p.setJointMotorControlArray(self.robot_id,range(4),p.POSITION_CONTROL,action)
        for _ in range(15):
            p.stepSimulation()
        obs = self.get_obs()

        hand_pos=p.getLinkState(self.robot_id,4)
        hand_pos=hand_pos[0]
        dist=np.linalg.norm(np.array(hand_pos)-self.target_pos)
        reward=-dist
        terminated=dist<0.02
        if terminated:
            reward+=300
        truncated=self.steps_since_reset>100
        
        
        return obs,reward,terminated,truncated,{}


