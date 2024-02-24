
'''
FILE THAT CONTAINS THE ONLY CLASS IMPLEMENTED IN THE PROJECT, A CLASS TO DESCRIBE
ALL THE DETAILS RELATIVE TO THE DISCRETIZATION OF THE RADIAL AXIS
'''

import numpy as np

class Grid:
    def __init__(self, number_of_points, r_max, r_min = 1e-5):
        self.r_min = r_min
        self.r_max = r_max
        self.number_of_points = number_of_points
        self.delta_r = (r_max - r_min) / (number_of_points - 1)
        self.r_values = np.linspace(r_min, r_max, number_of_points)