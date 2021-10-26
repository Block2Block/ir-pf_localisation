from functools import partial
from textwrap import fill
from numpy.core.fromnumeric import cumsum
from numpy.lib.financial import nper
from numpy.lib.function_base import append
from geometry_msgs.msg import Pose, PoseArray, Quaternion
from . pf_base import PFLocaliserBase
import math
import rospy

from . util import rotateQuaternion, getHeading
from random import random, uniform
from random import gauss
import random
import numpy as np
from time import time


class PFLocaliser(PFLocaliserBase):
       
    def __init__(self):
        # ----- Call the superclass constructor
        super(PFLocaliser, self).__init__()
        
        # ----- Set motion model parameters
        self.ODOM_ROTATION_NOISE = 5
        self.ODOM_TRANSLATION_NOISE = 0.5
        self.ODOM_DRIFT_NOISE = 0.5
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
        numofParticles = 200
        noise_parameter = 1

        rnd = random.normalvariate(0, 1)
 

        # Method 1 - Initilise around initialpose with gaussian noise
        for i in range(numofParticles):
            # adding gauss random noise to the 
            pose = Pose()
            pose.position.x = initialpose.pose.pose.position.x + (gauss(0,1) * 1)
            pose.position.y = initialpose.pose.pose.position.y + (gauss(0,1) * 1)
            pose.position.z = initialpose.pose.pose.position.z
            
            pose.orientation = rotateQuaternion(initialpose.pose.pose.orientation, math.radians(random.normalvariate(0, 30) + self.ODOM_ROTATION_NOISE)) # might need to change to guassian noise

            # add the partical to the PoseArray() object
            poseArray.poses.append(pose)

        # Method 2 - Initialise Random Uniformly
        # for i in range(numofParticles):
        #     pose = Pose()
        #     pose.position.x = uniform(-10,10)
        #     pose.position.y = uniform(-10,10)
        #     pose.position.z = initialpose.pose.pose.position.z
            
        #     pose.orientation = rotateQuaternion(initialpose.pose.pose.orientation, math.radians(uniform(0, 360))) # might need to change to guassian noise

        #     # add the partical to the PoseArray() object
        #     poseArray.poses.append(pose)
        
        # return the initailised particle in the form of a PoseArray() object
        return poseArray
    
    def update_particle_cloud(self, scan):
        """
        This should use the supplied laser scan to update the current
        particle cloud. i.e. self.particlecloud should be updated.
        
        :Args:
            | scan (sensor_msgs.msg.LaserScan): laser scan to use for update

         """
        
        scatterIndex = 0

        # Weight likelihood for each particles. p(z|x)
        weights = []
        for i in range(len(self.particlecloud.poses)):
            weights.append(self.sensor_model.get_weight(scan, self.particlecloud.poses[i]))

        # Normalise the list of weights - construct the outter ring
        normaliser = 1 / (sum(weights))
        normalisedWeights = []
        for i in range(len(weights)):
            normalisedWeights.append(weights[i] * normaliser)



        numofParticles = len(self.particlecloud.poses)
        updatedPoseList = []
        updatedPoseArray = PoseArray() # this will be the new PoseArray to set the self.particlecloud

        # Basic Roulette Wheel Resampling
        # updatedPoseList = np.random.choice(self.particlecloud.poses, len(self.particlecloud.poses), normalisedWeights)
        # for i in range(len(updatedPoseList)):
        #     updatedPoseArray.poses.append(updatedPoseList[i])

        # Another type of Rouletee Wheel thats
        # index = int(random.random() * numofParticles) # first index drawn at random
        # beta = 0    
        # maxWeight = max(normalisedWeights)
        # for i in range(numofParticles):
        #     beta = beta + random.random() * 2 * maxWeight  # this is the step we take
        #     while beta > normalisedWeights[index]: 
        #         beta = beta - normalisedWeights[index]
        #         index = (index + 1) % numofParticles
        #     updatedPoseList.append(self.particlecloud.poses[index])  # add this particle

        # Stratified Resamping


        # Systematic Resamping
        m = 2 * numofParticles # the number of particles we want, and we want the same number of particles as we initialised.
        cumSum = np.cumsum(normalisedWeights) # cumulative sum (outter ring)
        u = uniform(0, 1/m)
        # while u == 0:
        #     u = uniform(0, 1 / m)

        # [0.3, 0.7, 0.1] = 1
        # [0.3, 0.9, 1] = cumSum

        # u = 0.1
        # u = 0.2
        # u = 0.3
        # u = 0.4
        # u = 0.5
        # u = 0.6
        # u = 0.7
        # u = 0.8
        # u = 0.9
        # u = 1

        # [p1, p2, p3]


        # [p1, p1, p1, p2, p2, p2, p2, p2, p2, p3]

        i = 0
        j = 0

        while j < m:
            if (u <= cumSum[i]):
                updatedPoseList.append(self.particlecloud.poses[i])
                j = j + 1
                u = u + 1 / m
            else:
                i = i + 1

        for i in range(len(updatedPoseList)):
            updatedPose = Pose()
            updatedPose.position.x = updatedPoseList[i].position.x
            updatedPose.position.y = updatedPoseList[i].position.y
            updatedPose.orientation = updatedPoseList[i].orientation

            updatedPoseArray.poses.append(updatedPose)

        print(len(updatedPoseArray.poses))
        
        # reduce back to original particle size
        while len(updatedPoseArray.poses) > numofParticles:
            updatedPoseArray.poses.pop(random.randrange(len(updatedPoseArray.poses)))
        
        print(len(updatedPoseArray.poses))

        print("============================================")
        
        # Systematic Resamping 2

        # m = len(self.particlecloud.poses) # the number of particles we want, and we want the same number of particles as we initialised.
        # position = (random.random() + np.arange(m)) / m
        # cumSum = np.cumsum(normalisedWeights)
        # #or  cumSum = np.cumsum(normalisedWeights)
        
        # i, j = 0, 0

        # while i < m:
        #     if (position[i] < cumSum[j]):
        #         updatedPoseList.append(self.particlecloud.poses[j])
        #         i = i + 1
        #     else:
        #         j = j + 1

        # #low variance sampling
        # #implementing based on the algorithm table 4.4 of Probabilistic Robotics
        # #resampledPose = []
        # #resampledPoseArray = PoseArray()
        # M = len(self.particlecloud.poses)
        # r = random.uniform(0,1/M)
        # c = weights[1]
        # i = 1
        # for m in range(1,M):
        #     U = r + (m-1)*(1/M)
        #     while U>c:
        #         i += 1
        #         c += weights[i]
        #     updatedPoseList.append(self.particlecloud.poses[i])

        #Add noise - need to make this only run in certain condition, such as when the speard of the particles is high
        # for i in range(len(updatedPoseList)):
        #     updatedPose = Pose()
        #     updatedPose.position.x = gauss(updatedPoseList[i].position.x, 0.3)
        #     updatedPose.position.y = gauss(updatedPoseList[i].position.y, 0.3)
        #     updatedPose.orientation = rotateQuaternion(updatedPoseList[i].orientation, math.radians(gauss(math.degrees(getHeading(updatedPoseList[i].orientation)),1)))

        #     updatedPoseArray.poses.append(updatedPose)

        # Scatter the particles
        # if (scatterIndex % 10 == 0): # Scatter the particles every 5 iterations.
        #     for i in range(5):
        #         scatterPose = Pose()
        #         scatterPose.position.x = random.uniform(-5, 5)
        #         scatterPose.position.y = random.uniform(-5, 5)

        #         q_orig = [0,0,0,1]
        #         q_orig_msg = Quaternion(q_orig[0], q_orig[1], q_orig[2], q_orig[3])
        #         scatterPose.orientation = rotateQuaternion(q_orig_msg, math.radians(np.random.uniform(0, 360)))

        #         updatedPoseArray.poses.pop(random.randrange(len(updatedPoseArray.poses)))
        #         updatedPoseArray.poses.append(scatterPose)

        #     scatterIndex = scatterIndex + 1

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
        
        numofParticles = len(self.particlecloud.poses)

        # # Remove outlier first method -----------------------------------------------------------------------------
        # outlierFreePoseArrray = PoseArray()
        # #using z score to eliminate outliers in the particle cloud.
        # # Since we are eliminating it based on the position the orientation is not considered in our calculations.
        # meanPositionX = np.mean([self.particlecloud.poses[i].position.x for i in range(len(self.particlecloud.poses))])
        # meanPositionY = np.mean([self.particlecloud.poses[i].position.y for i in range(len(self.particlecloud.poses))])
        
        # stdPositionX = np.std([self.particlecloud.poses[i].position.x for i in range(len(self.particlecloud.poses))])
        # stdPositionY = np.std([self.particlecloud.poses[i].position.y for i in range(len(self.particlecloud.poses))])
        
        
        # #setting the threshold as 3 as of now. It should be changed depending on how dense/tight the cluster is needed.
        # zs_threshold = 3
        # #calculation of average z score
        # sumofZS_x, sumofZS_y = 0, 0
        # for cnt in range(len(self.particlecloud.poses)):
        #     zs_x = (self.particlecloud.poses[cnt].position.x - meanPositionX)/stdPositionX
        #     zs_y = (self.particlecloud.poses[cnt].position.y - meanPositionY)/stdPositionY
        #     sumofZS_x = sumofZS_x + zs_x
        #     sumofZS_y = sumofZS_y + zs_y
        
        # meanZS_x = sumofZS_x/numofParticles
        # meanZS_y = sumofZS_y/numofParticles
        # avg_ZS = (meanZS_y + meanZS_x)/2
        
        # #dropping outlier particles
        # for cnt in range(len(self.particlecloud.poses)):
        #     zs_particlex = (self.particlecloud.poses[cnt].position.x - meanPositionX)/stdPositionX
        #     zs_particley = (self.particlecloud.poses[cnt].position.y - meanPositionY)/stdPositionY
        #     zs_avg_particle = (zs_particlex + zs_particley)/2
        #     if zs_avg_particle <= avg_ZS:
        #         outlierFreePoseArrray.poses.append(self.particlecloud.poses[cnt])
        
        # # returning outlier free updated pose array
        # #self.particlecloud = outlierFreePoseArrray

        # sumofX = 0
        # sumofY = 0
        # sumofQX = 0
        # sumofQY = 0
        # sumofQZ = 0
        # sumofQW = 0

        # # Summing the particles
        # for i in range(len(outlierFreePoseArrray.poses)):
        #     sumofX = sumofX + outlierFreePoseArrray.poses[i].position.x
        #     sumofY = sumofY + outlierFreePoseArrray.poses[i].position.y

        #     sumofQX = sumofQX + outlierFreePoseArrray.poses[i].orientation.x
        #     sumofQY = sumofQY + outlierFreePoseArrray.poses[i].orientation.y
        #     sumofQZ = sumofQZ + outlierFreePoseArrray.poses[i].orientation.z
        #     sumofQW = sumofQW + outlierFreePoseArrray.poses[i].orientation.w

        # estimateAveragePose = Pose()
        # estimateAveragePose.position.x = sumofX / numofParticles
        # estimateAveragePose.position.y = sumofY / numofParticles
        # estimateAveragePose.orientation.x = sumofQX / numofParticles
        # estimateAveragePose.orientation.y = sumofQY / numofParticles
        # estimateAveragePose.orientation.z = sumofQZ / numofParticles
        # estimateAveragePose.orientation.w = sumofQW / numofParticles

        # Basic Averge Estimate --------------------------------------------------------------------------
        # These are to store the sum of particles, so we can than take the average, to get the estimate pose.
        sumofX = 0
        sumofY = 0
        sumofQX = 0
        sumofQY = 0
        sumofQZ = 0
        sumofQW = 0

        # Summing the particles
        for i in range(len(self.particlecloud.poses)):
            sumofX = sumofX + self.particlecloud.poses[i].position.x
            sumofY = sumofY + self.particlecloud.poses[i].position.y

            sumofQX = sumofQX + self.particlecloud.poses[i].orientation.x
            sumofQY = sumofQY + self.particlecloud.poses[i].orientation.y
            sumofQZ = sumofQZ + self.particlecloud.poses[i].orientation.z
            sumofQW = sumofQW + self.particlecloud.poses[i].orientation.w

        # Averaging the particles
        estimateAveragePose = Pose()
        estimateAveragePose.position.x = sumofX / numofParticles
        estimateAveragePose.position.y = sumofY / numofParticles
        estimateAveragePose.orientation.x = sumofQX / numofParticles# I think I forgot the averge it here???
        estimateAveragePose.orientation.y = sumofQY / numofParticles
        estimateAveragePose.orientation.z = sumofQZ / numofParticles
        estimateAveragePose.orientation.w = sumofQW / numofParticles

        return estimateAveragePose