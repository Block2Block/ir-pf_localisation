from numpy.lib.financial import nper
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

        noise_parameter = 0.5 # what is a good noise_parameter? are there any pros and cons for having more noise?
        # I think the answser: it is good to have more noise then less, since it this mean we will have a higher chance of capturing the position
        # of the robot, whereas if we have less noise, there is a chance that we never got a particle which is close to where the robot is located.

        # generating particles - how many particles to generate? The more particles means higher chance of capturing the postion of the robot,
        # less particles means that we might not have particle which is close to where the robot it located.
        for i in range(20):

            # adding gauss random noise to the 
            pose = Pose()
            pose.position.x = initialpose.pose.pose.position.x + (gauss(0,1) * noise_parameter)
            pose.position.y = initialpose.pose.pose.position.y + (gauss(0,1) * noise_parameter)
            pose.position.z = initialpose.pose.pose.position.z
            
            uniform_ran_quat = rotateQuaternion(initialpose.pose.pose.orientation, math.radians(np.random.uniform(0, 360))) # might need to change to guassian noise
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
        updatedPoseList = []
        updatedPoseArray = PoseArray() # this will be the new PoseArray to set the self.particlecloud

        # Weight likelihood for each particles. p(z|x)
        weights = []
        for i in range(len(self.particlecloud.poses)):
            weights.append(self.sensor_model.get_weight(scan, self.particlecloud.poses[i]))

        # Normalise the list of
        numOfWeights = len(self.particlecloud.poses)
        normaliser = 1 / float(len(self.particlecloud.poses))
        maxWeight = 0.0
        minWeight = 0.0
        normalisedWeights = []
        
        for i in range(len(weights)):
            normalisedWeights.append(weights[i] * normaliser)

        # Resample the particles - Roulette Wheel Method - might need to implement this, instead of using this numpy function
        updatedPoseList = np.random.choice(self.particlecloud.poses, len(self.particlecloud.poses), weights)
        
        # Add noise
        for i in range(len(updatedPoseList)):
            updatedPose = Pose()
            updatedPose.position.x = gauss(updatedPoseList[i].position.x, 0.2)
            updatedPose.position.y = gauss(updatedPoseList[i].position.y, 0.2)
            updatedPose.orientation = rotateQuaternion(updatedPoseList[i], math.radians(np.random.uniform(0, 10)))

            updatedPoseArray.poses.append(updatedPose)

        self.particlecloud = updatedPoseArray

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

        # These are to store the sum of particles, so we can than take the average, to get the estimate pose.
        sumofX = 0
        sumofY = 0
        sumofQX = 0
        sumofQY = 0
        sumofQZ = 0
        sumofQW = 0

        numOfParticles = len(self.particlecloud.poses)

        # Do we need this initilisation it? Probably not.
        # averageQuat = Quaternion()

        sumofOrientation = 0

        for i in range(len(self.particlecloud.poses)):
            sumofX = sumofX + self.particlecloud.poses[i].position.x
            sumofY = sumofY + self.particlecloud.poses[i].position.y

            sumofQX = sumofQX + self.particlecloud.poses[i].orientation.x
            sumofQY = sumofQY + self.particlecloud.poses[i].orientation.y
            sumofQZ = sumofQZ + self.particlecloud.poses[i].orientation.z
            sumofQW = sumofQW + self.particlecloud.poses[i].orientation.w

        averagePose = Pose()
        averagePose.position.x = sumofX / numOfParticles
        averagePose.position.y = sumofY / numOfParticles
        averagePose.orientation.x = sumofQX
        averagePose.orientation.y = sumofQY
        averagePose.orientation.z = sumofQZ
        averagePose.orientation.w = sumofQW

        return averagePose