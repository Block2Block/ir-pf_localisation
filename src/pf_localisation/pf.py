from geometry_msgs.msg import Pose, PoseArray, Quaternion
from . pf_base import PFLocaliserBase
import math
import rospy

from . util import rotateQuaternion, getHeading
from random import random
from random import gauss
import numpy as np
from time import time


class PFLocaliser(PFLocaliserBase):
       
    def __init__(self):
        # ----- Call the superclass constructor
        super(PFLocaliser, self).__init__()
        
        # ----- Set motion model parameters
        self.ODOM_ROTATION_NOISE = 1
        self.ODOM_TRANSLATION_NOISE = 1
        self.ODOM_DRIFT_NOISE = 1
        # ----- Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 20     # Number of readings to predict
        
       
    def initialise_particle_cloud(self, initialpose):
        """
        Set particle cloud to initialpose plus noise

        Called whenever an initialpose message is received (to change the
        starting location of the robot), or a new occupancy_map is received.
        self.particlecloud can be initialised here. Initial pose of the robot
        is also set here.
        
        :Args:
            | initialpose: the initial pose estimate
        :Return:
            | (geometry_msgs.msg.PoseArray) poses of the particles
        """
        poseArray = PoseArray()

        noise_parameter = 0.5

        # initiallise 20 particles
        for i in range(20):

            # generate gaussian random value of each particle
            pose = Pose()
            pose.position.x = initialpose.position.x + (gauss(0,1) * noise_parameter)
            pose.position.y = initialpose.position.y + (gauss(0,1) * noise_parameter)
            pose.position.z = initialpose.position.z
            
            uniform_ran_quat = rotateQuaternion(initialpose.orientation, math.radians(np.random.uniform(0, 360)))
            pose.orientation.x = uniform_ran_quat.x 
            pose.orientation.y = uniform_ran_quat.y
            pose.orientation.z = uniform_ran_quat.z # need to figure out how to add noise to this quat thing
            pose.orientation.w = uniform_ran_quat.w

            # add the partical to the PoseArray() object
            poseArray.poses.append(pose)
        
        # return the initailised particle in the form of a PoseArray() object
        return poseArray
 
    
    def update_particle_cloud(self, scan):
        """
        This should use the supplied laser scan to update the current
        particle cloud. i.e. self.particlecloud should be updated.
        
        :Args:
            | scan (sensor_msgs.msg.LaserScan): laser scan to use for update

         """
        pass

    def estimate_pose(self):
        """
        This should calculate and return an updated robot pose estimate based
        on the particle cloud (self.particlecloud).
        
        Create new estimated pose, given particle cloud
        E.g. just average the location and orientation values of each of
        the particles and return this.
        
        Better approximations could be made by doing some simple clustering,
        e.g. taking the average location of half the particles after 
        throwing away any which are outliers

        :Return:
            | (geometry_msgs.msg.Pose) robot's estimated pose.
         """
        pass
