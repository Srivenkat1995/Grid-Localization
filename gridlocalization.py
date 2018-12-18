#!/usr/bin/env python

import roslib
import rospy
import sys
import rosbag
import numpy as np
import math
from tf.transformations import euler_from_quaternion


#### Intialize Grid ###
### Given the total area coverage as 7m*7m ####
### Each grid cell is 0.2m*0.2m , so the no of squares is 1225 -----> Each row having 35 squares and column having 35 squares #####

grid_3d_probablity_values = np.zeros((35,35,8))


# Making the range to -180 to 180 .. #################

def convert_orientation_to_within_range(orientation):

    
    if orientation > 180:

        return (orientation - 360)

    elif orientation < -180:

        return (orientation + 360)

    else:

        return orientation    


def convert_continuous_grid(grid_position_x, grid_position_y, grid_oreintation):

    return ((grid_position_x*0.2 + 0.1),(-180 + grid_oreintation * 45),(grid_position_y * 0.2 + 0.1))


def new_model_calculation(x1,y1,orientation1,x2,y2,orientation2):

    new_translation = math.sqrt((x1-x2) ** 2 + (y1-y2) ** 2)

    new_orientation = np.degrees(np.arctan2((y1-y2),(x1-x2)))

    rotation1 = convert_orientation_to_within_range((orientation1 - new_orientation))

    rotation2 = convert_orientation_to_within_range((new_orientation - orientation2))
    
    return rotation2,new_translation,rotation1


def gaussian_distribution(mean,covariance):

    numerator = np.power(np.e, -1.0 * (mean ** 2) / (2 * (covariance ** 2)))

    denominator = (math.sqrt(2 * np.pi) * covariance)

    return numerator/denominator



def motion_model(rotation1, translation ,rotation2):

    global grid_3d_probablity_values, copy_of_grid_3d_probablity_values

    copy_of_grid_3d_probablity_values = np.copy(grid_3d_probablity_values)

    x,y,z = grid_3d_probablity_values.shape

    probablity = 0

    for i in np.arange(x):

        for j in np.arange(y):

            for k in np.arange(z):

                if copy_of_grid_3d_probablity_values[i,j,k] < 0.1:

                    continue

                for a in np.arange(x):

                    for b in np.arange(y):

                        for c in np.arange(z):

                            x1_grid, orientation1_grid, y1_grid = convert_continuous_grid(a,b,c)

                            x2_grid, orientation2_grid, y2_grid = convert_continuous_grid(i,j,k)    

                            new_rotation1, new_translation, new_rotation2 = new_model_calculation(x1_grid,y1_grid,orientation1_grid,x2_grid,y2_grid,orientation2_grid)

                            probablity_rotation_1 =  gaussian_distribution((new_rotation1 - rotation1), 45)

                            probablity_translation = gaussian_distribution((new_translation - translation), 0.1)

                            probablity_rotation_2 = gaussian_distribution((new_rotation2 - rotation2), 45)

                            combined_probablity = probablity_rotation_1 * probablity_rotation_2 * probablity_translation

                            new_probablity_w_r_t_grid = copy_of_grid_3d_probablity_values[i,j,k] * combined_probablity

                            grid_3d_probablity_values[a,b,c] = grid_3d_probablity_values[a,b,c] + new_probablity_w_r_t_grid

                            probablity = probablity + new_probablity_w_r_t_grid
    
    grid_3d_probablity_values = grid_3d_probablity_values/probablity

    index_of_max_value = np.argmax(grid_3d_probablity_values)

    x,y, theta = calculate_grid_box_from_index(index_of_max_value,grid_3d_probablity_values)

    return (x,y,theta)
     

def new_measurement_calculation(x,y,orientation, landmark_location_position_map,tag_no):

    translation = math.sqrt((landmark_location_position_map.item((tag_no,0)) - x ) ** 2 + (landmark_location_position_map.item((tag_no,1)) - y ) ** 2)

    orient = np.degrees(np.arctan2((landmark_location_position_map.item((tag_no,1)) - y ),(landmark_location_position_map.item((tag_no,0)) - x)))

    rotation = convert_orientation_to_within_range(orient -orientation)

    return rotation,translation


def measurement_model(bearing,ranges,tag_no, landmark_location_position_map):

    global grid_3d_probablity_values, copy_of_grid_3d_probablity_values

    copy_of_grid_3d_probablity_values = np.array(grid_3d_probablity_values)

    probablity = 0 

    x,y,z = grid_3d_probablity_values.shape

    for i in np.arange(x):

        for j in np.arange(y):

            for k in np.arange(z):

                x_grid, orientation_grid , y_grid = convert_continuous_grid(i,j,k)

                new_rotation, new_translation = new_measurement_calculation(x_grid,y_grid,orientation_grid,landmark_location_position_map,tag_no)

                probablity_rotation = gaussian_distribution((new_rotation - bearing), 45)

                probablity_translation = gaussian_distribution((new_translation - ranges), 0.1)

                combined_probablity = probablity_rotation * probablity_translation

                new_probablity_w_r_t_grid = copy_of_grid_3d_probablity_values[i,j,k] * combined_probablity

                grid_3d_probablity_values[i,j,k] = new_probablity_w_r_t_grid

                probablity += new_probablity_w_r_t_grid

    grid_3d_probablity_values = grid_3d_probablity_values/probablity

    index = np.argmax(grid_3d_probablity_values)

    x,y,theta = calculate_grid_box_from_index(index,grid_3d_probablity_values)

    return (x,y,theta)

def calculate_grid_box_from_index(index,grid_3d_probablity_values):

    x,y,z = grid_3d_probablity_values.shape

    index_y = index / z

    index_x = index_y / y

    return ((index_x % x),(index_y % y),(index % z))


if __name__ == '__main__':

    #global grid_3d_probablity_values

    rospy.init_node('Grid_Localization')
    
    name_of_the_bag = rosbag.Bag("/home/first/catkin_ws/src/lab4/grid.bag",'r')


    ##### Intialize the landmark positions #######################

    landmark_location_position_map = np.matrix([[1.25,5.25],[1.25,3.25],[1.25,1.25],[4.25,1.25],[4.25,3.25],[4.25,5.25]])

    ##### Intialize the robot location ########################### (Given as  (12,28,3) -----> in the array (11,27,2))

    grid_3d_probablity_values[11,27,2] = 1 

    file=  open('trajectory.txt','w')
    #mark_landmarks_in_location(landmark_location_position_map


    for topic, msg, time_stamp in name_of_the_bag.read_messages(topics=['Movements', 'Observations']):
        
        if topic == 'Movements':
            
                print ("Movements")
            
                rotation1 = np.degrees((euler_from_quaternion([msg.rotation1.x,msg.rotation1.y,msg.rotation1.z,msg.rotation1.w]))[2])
            
                translation = msg.translation
            
                rotation2 = np.degrees((euler_from_quaternion([msg.rotation2.x,msg.rotation2.y,msg.rotation2.z,msg.rotation2.w]))[2])
            
                x,y,theta = motion_model(rotation1, translation, rotation2)

                print(x,y,theta)

        
        else:
    
            print("Observations")
                
            bearing = np.degrees((euler_from_quaternion([msg.bearing.x,msg.bearing.y,msg.bearing.z,msg.bearing.w]))[2])
    
            ranges = msg.range

            tag_no = msg.tagNum

            x,y,theta = measurement_model(bearing,ranges, tag_no, landmark_location_position_map)

            file.write(str(x+1) + ',' + str(y+1) + ',' + str(theta+1)+ '\n' )

            print(x,y,theta)

            
    name_of_the_bag.close()

    file.close()
            
