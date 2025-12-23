import pybullet as p
import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image
import time
from collections import deque






class Env(gym.Env):
    
    def __init__(self,render_mode="none"):
        super().__init__()
        self.joint_indices = [0, 1, 2, 3]
        self.max_vel=40
   
        self.ll = np.array([-3.1416, -1.5708, -1.0, 0.0])
        self.ul = np.array([3.1416, 1.5708, 3.1416, 1.5])
        

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.render_mode=render_mode
        self.frame_stack = deque(maxlen=4)
        self.client=p.connect(p.GUI if render_mode=="human" else p.DIRECT)
        
        urdf_path = r"C:\Users\User\Desktop\roarm_description\urdf\roarm_description.urdf"
        
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32),
            "image": spaces.Box(low=-np.inf, high=np.inf, shape=(4,64,64), dtype=np.float32)
            })
        
        self.robot_id=p.loadURDF(urdf_path,useFixedBase=True,physicsClientId=self.client,flags=p.URDF_MAINTAIN_LINK_ORDER | p.URDF_USE_SELF_COLLISION)
        self.point_vis=p.createVisualShape(p.GEOM_SPHERE,0.04,rgbaColor=[1,0,0,1],physicsClientId=self.client)
        self.point=p.createMultiBody(0.1,baseVisualShapeIndex=self.point_vis,basePosition=[0,0,0],physicsClientId=self.client)
        self.camera_idx = -1
        
        for i in range(p.getNumJoints(self.robot_id,self.client)):
            joint_info = p.getJointInfo(self.robot_id, i,physicsClientId=self.client)
            if joint_info[12].decode('utf-8') == "camera_link":
                self.camera_idx = i
        self.hand_idx = -1
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            if info[12].decode('utf-8') == "hand_tcp":
                self.hand_idx = i
            p.changeDynamics(self.robot_id, i, 
                     linearDamping=0.04, 
                     angularDamping=0.04, 
                     jointDamping=0.5)
        
        p.setAdditionalSearchPath(os.path.dirname(__file__))

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client) 
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=self.client)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=self.client)
        p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0, physicsClientId=self.client)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.client)
        
        if render_mode=="human":
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            self.action_sliders = []
            for i in range(3):
      
                s_id = p.addUserDebugParameter(f"Joint {i+1} Velocity", -3.14, 3.14, 0)
                
                self.action_sliders.append(s_id)
            self.toggle_id=p.addUserDebugParameter("Use sliders?",0,1,0)
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1, physicsClientId=self.client)
        else:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=self.client)
            p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0, physicsClientId=self.client)
        self.prev_hand_pos= p.getLinkState(self.robot_id, self.hand_idx)[0]
        self.collision_dist=0.13
        
    def get_obs(self):
        obs=np.zeros((9,),dtype=np.float32)
        for i in range(p.getNumJoints(self.robot_id,physicsClientId=self.client)-3):
            obs[i]=p.getJointState(self.robot_id,i,physicsClientId=self.client)[0]/np.pi
            obs[i+4]=p.getJointState(self.robot_id,i,physicsClientId=self.client)[1]/self.max_vel
        obs[8]=self.get_num_of_ball_pixels()/(64*64)
        img_stack = np.stack(self.frame_stack, axis=0).astype(np.float32)
        img_stack /= 255.0
        obs={
            "obs": obs,
            "image": img_stack
        }
        return obs
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_since_reset = 0
        

        spawn_pos=np.array((99999, 9999, 999999))
        while self.is_out_of_bounds(spawn_pos):
            for i, joint_idx in enumerate(self.joint_indices):
                angle = np.random.uniform(self.ll[i], self.ul[i])
                p.resetJointState(self.robot_id, joint_idx, angle, physicsClientId=self.client)
            

            state = p.getLinkState(self.robot_id, self.camera_idx, computeForwardKinematics=True, physicsClientId=self.client)
            pos_cam, orn_cam = state[0], state[1]
            rot_mat = p.getMatrixFromQuaternion(orn_cam, physicsClientId=self.client)
            
         
            forward_vec = np.array([rot_mat[2], rot_mat[5], rot_mat[8]])
            spawn_pos = np.array(pos_cam) + (forward_vec * 0.25) + np.random.uniform(-0.1, 0.1, 3)
            
        if np.random.random()<0.33:
            spawn_pos=np.array((99999, 9999, 999999))
            while self.is_out_of_bounds(spawn_pos):
                spawn_pos=np.random.uniform([-0.3,-0.3,0], [0.3,0.3,0.3], 3)
        p.resetBasePositionAndOrientation(self.point, spawn_pos, [0,0,0,1], self.client)
 
    
        self.target_pos=spawn_pos
       
        img = self.get_image()
        img = img[0, :, :, :3]
        gray = ((img[:, :, 0] > 150) & (img[:, :, 1] < 20)).astype(np.uint8) * 255
        self.frame_stack.clear()
        for _ in range(4):
            self.frame_stack.append(gray)
            
        self.prev_hand_pos = p.getLinkState(self.robot_id, self.hand_idx)[0]
        return self.get_obs(), {}
    def step(self,action):
        self.steps_since_reset+=1
        
        self.prev_hand_pos= p.getLinkState(self.robot_id, self.hand_idx)[0]

        self.max_vel
        force=30

        target_vel = np.clip(action * self.max_vel,
                     -self.max_vel, self.max_vel)

        current_positions = np.array([p.getJointState(self.robot_id, i)[0] 
                                    for i in self.joint_indices])
        for i in range(len(self.joint_indices)):
            if (current_positions[i] <= self.ll[i] and target_vel[i] < 0) or \
            (current_positions[i] >= self.ul[i] and target_vel[i] > 0):
                target_vel[i] = 0
        p.setJointMotorControlArray(
            self.robot_id,
            self.joint_indices,
            p.VELOCITY_CONTROL,
            targetVelocities=target_vel,
            forces=[force]*len(self.joint_indices),
            physicsClientId=self.client
        )
        

        
        if self.render_mode=="human":
            
            for _ in range(8):
                time.sleep(1/(60*8))
                p.stepSimulation(self.client)
            
        else:
            for _ in range(8):
                p.stepSimulation(self.client)
       
        hand_pos = p.getLinkState(self.robot_id, self.hand_idx)[0]
        
        
        prev_dist = np.linalg.norm(np.array(self.prev_hand_pos) - self.target_pos)
        curr_dist = np.linalg.norm(hand_pos - self.target_pos)

        
        
        
        img=self.get_image()
        red_channel = img[0, :, :, 0] 
        green_channel = img[0, :, :, 1] 
        red_pixels = np.sum((red_channel > 100) & (green_channel<50))
        black_pixels=np.sum((red_channel <50) & (green_channel<50))
        if red_pixels>0 and self.render_mode=="human":
            #print(red_pixels/(64*64)*100)
            pass
        if black_pixels>0 and self.render_mode=="human":
            print(11111111111111111111111111111111111)

        
        gray = img[0, :, :, :3]
        gray = ((gray[:, :, 0] > 150) & (gray[:, :, 1] < 20)).astype(np.uint8) * 255
        self.frame_stack.append(gray)
        terminated=curr_dist<self.collision_dist



       
        reward = 0
        reward += np.clip(red_pixels / (64*64), 0.0, 0.05)
        reward-=np.clip(black_pixels / (64*64), 0.0, 0.05)

        if curr_dist < 0.25:
            reward += 3 
        if curr_dist < 0.15:
            reward += 6
        if terminated:
            reward += 500.0 
        truncated=self.steps_since_reset>500
        if truncated:
            reward-=500.0

        return self.get_obs(), reward, terminated, truncated, {}
    def get_image(self):
        state = p.getLinkState(self.robot_id, self.camera_idx, computeForwardKinematics=True,physicsClientId=self.client)
        pos=state[0]
        orn=state[1]
        rot_mat = p.getMatrixFromQuaternion(orn,physicsClientId=self.client)
        forward_vec = [rot_mat[2], rot_mat[5], rot_mat[8]]
        target = [pos[0] + forward_vec[0], pos[1] + forward_vec[1], pos[2] + forward_vec[2]]
        up_vec = [rot_mat[0], rot_mat[3], rot_mat[6]]
        up = [pos[0] + up_vec[0], pos[1] + up_vec[1], pos[2] + up_vec[2]]
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=pos,
            cameraTargetPosition=target,
            cameraUpVector=up_vec,
            physicsClientId=self.client
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=68.5, aspect=1, nearVal=0.01, farVal=5.0, physicsClientId=self.client
        )
        (_, _, rgb, _, _) = p.getCameraImage(
            width=64, height=64,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            flags=p.ER_NO_SEGMENTATION_MASK | p.ER_USE_PROJECTIVE_TEXTURE,
            physicsClientId=self.client
        )
        if self.render_mode == "human":
            p.addUserDebugLine(pos, target, [1, 1, 0], lineWidth=2, lifeTime=0.1)
            p.addUserDebugLine(pos, up, [0, 1, 0], lineWidth=2, lifeTime=0.1)
        rgb = np.array(rgb, dtype=np.uint8).reshape(64, 64, 4)
        
         
        
        
        return rgb[np.newaxis, :, :]
    def get_num_of_ball_pixels(self):
        img=self.get_image()
        red_channel = img[0, :, :, 0] 
        green_channel = img[0, :, :, 1] 
        red_pixels = np.sum((red_channel > 100) & (green_channel<50))
        return red_pixels
    def is_out_of_bounds(self, pos):
        """Based on URDF: L2(0.236) + L3(0.215) + BaseOffset"""
        dist_from_base = np.linalg.norm(pos)
        horizontal_dist = np.linalg.norm(pos[:2])

        MAX_SAFE_REACH = 0.42  
        MIN_SAFE_REACH = 0.12  
        
        return (dist_from_base > MAX_SAFE_REACH or 
                horizontal_dist < MIN_SAFE_REACH or 
                pos[2] < 0.05 or 
                pos[2] > 0.45)

if __name__ == '__main__':
    env = Env(render_mode="human")
    env.reset()
    
    while True:

        current_joints = p.getJointStates(env.robot_id, range(4))
        current_positions = np.array([state[0] for state in current_joints])
        

        slider_targets = np.zeros(4)
        if p.readUserDebugParameter(env.toggle_id)==1:
            for i in range(3):
                slider_targets[i] = p.readUserDebugParameter(env.action_sliders[i])
            

            
            action_needed = ((slider_targets - current_positions))       

            action_clipped = np.clip(action_needed, -3.14, 3.14)
        else:
            action_clipped=np.zeros((4,))
        

        obs, reward, terminated, truncated, _ = env.step(action_clipped)
        
        
        