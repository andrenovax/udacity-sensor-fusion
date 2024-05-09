# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        pass

    def F(self):
        ############
        # TODO Step 1: implement and return system matrix F
        ############

        return np.matrix([
            [1, 0, 0, params.dt, 0, 0],
            [0, 1, 0, 0, params.dt, 0],
            [0, 0, 1, 0, 0, params.dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])
        
        ############
        # END student code
        ############ 

    def Q(self):
        ############
        # TODO Step 1: implement and return process noise covariance Q
        ############

        q3 = 1/3*params.dt**3*params.q
        q2 = 1/2*params.dt**2*params.q
        q1 = params.dt*params.q

        return np.matrix([
            [q3, 0, 0, q2, 0, 0],
            [0, q3, 0, 0, q2, 0],
            [0, 0, q3, 0, 0, q2],
            [q2, 0, 0, q1, 0, 0],
            [0, q2, 0, 0, q1, 0],
            [0, 0, q2, 0, 0, q1],
        ])
        
        ############
        # END student code
        ############ 

    def predict(self, track):
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############
        F = self.F()

        x = F * track.x
        P = F * track.P * np.transpose(F) + self.Q()

        track.set_x(x)
        track.set_P(P)

        ############
        # END student code
        ############ 

    def update(self, track, meas):
        ############
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############

        H = meas.sensor.get_H(track.x)
        S = self.S(track, meas, H)
        K = track.P * np.transpose(H) * np.linalg.inv(S)
        gamma = self.gamma(track, meas)
        x = track.x + K * gamma
        P = (np.identity((params.dim_state)) - K * H) * track.P

        track.set_x(x)
        track.set_P(P)

        ############
        # END student code
        ############ 
        track.update_attributes(meas)
    
    def gamma(self, track, meas):
        ############
        # TODO Step 1: calculate and return residual gamma
        ############

        h_x = meas.sensor.get_hx(track.x)
        gamma = meas.z - h_x

        return gamma
        
        ############
        # END student code
        ############ 

    def S(self, track, meas, H):
        ############
        # TODO Step 1: calculate and return covariance of residual S
        ############

        S = H * track.P * np.transpose(H) + meas.R

        return S
        
        ############
        # END student code
        ############ 