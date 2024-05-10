# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
from scipy.stats.distributions import chi2
from scipy.optimize import linear_sum_assignment

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import misc.params as params 

class Association:
    '''Data association class with single nearest neighbor association and gating based on Mahalanobis distance'''
    def __init__(self):
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_meas = []
        
    def associate(self, track_list, meas_list, KF):
             
        ############
        # TODO Step 3: association:
        # - replace association_matrix with the actual association matrix based on Mahalanobis distance (see below) for all tracks and all measurements
        # - update list of unassigned measurements and unassigned tracks
        ############

        max_value = params.max_P + 1 if params.association_method == 'gnn' else np.inf
        
        # the following only works for at most one track and one measurement
        self.association_matrix = np.full((len(track_list), len(meas_list)), max_value)

        for track_index, track in enumerate(track_list):
            for meas_index, meas in enumerate(meas_list):
                dist = self.MHD(track, meas, KF)
                if self.gating(dist, meas.sensor):
                    self.association_matrix[track_index][meas_index] = dist

        self.unassigned_tracks = list(range(len(track_list))) # reset lists
        self.unassigned_meas = list(range(len(meas_list)))

        ############
        # END student code
        ############ 
                
    def get_closest_track_and_meas(self):
        ############
        # TODO Step 3: find closest track and measurement:
        # - find minimum entry in association matrix
        # - delete row and column
        # - remove corresponding track and measurement from unassigned_tracks and unassigned_meas
        # - return this track and measurement
        ############

        # the following only works for at most one track and one measurement
        min_index = np.unravel_index(np.argmin(self.association_matrix), self.association_matrix.shape)
        update_track_index, update_meas_index = min_index

        update_track = self.unassigned_tracks[update_track_index]
        update_meas = self.unassigned_meas[update_meas_index]

        # remove from list
        self.unassigned_tracks.pop(update_track_index)
        self.unassigned_meas.pop(update_meas_index)

        A = np.delete(self.association_matrix, update_track_index, axis=0)
        self.association_matrix = np.delete(A, update_meas_index, axis=1)

        ############
        # END student code
        ############ 
        return update_track, update_meas     

    def gating(self, MHD, sensor): 
        ############
        # TODO Step 3: return True if measurement lies inside gate, otherwise False
        ############

        return MHD <= chi2.ppf(params.gating_threshold, sensor.dim_meas)
        
        ############
        # END student code
        ############ 
        
    def MHD(self, track, meas, KF):
        ############
        # TODO Step 3: calculate and return Mahalanobis distance
        ############

        gamma = KF.gamma(track, meas)

        H = meas.sensor.get_H(track.x)
        S = KF.S(track, meas, H)

        return np.transpose(gamma) * np.linalg.inv(S) * gamma
        
        ############
        # END student code
        ############ 
    def associate_and_update(self, *args, **kwargs):
        if params.association_method == 'gnn':
            return self.associate_and_update_gnn(*args, **kwargs)

        return self.associate_and_update_nn(*args, **kwargs)
    
    def associate_and_update_nn(self, manager, meas_list, KF):
        # associate measurements and tracks
        self.associate(manager.track_list, meas_list, KF)
    
        # update associated tracks with measurements
        while self.association_matrix.shape[0]>0 and self.association_matrix.shape[1]>0:
            
            # search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas()
            if np.isnan(ind_track):
                print('---no more associations---')
                break
            track = manager.track_list[ind_track]
            
            # check visibility, only update tracks in fov    
            if not meas_list[0].sensor.in_fov(track.x):
                continue
            
            # Kalman update
            print('update track', track.id, 'with', meas_list[ind_meas].sensor.name, 'measurement', ind_meas)
            KF.update(track, meas_list[ind_meas])
            
            # update score and track state 
            manager.handle_updated_track(track)
            
            # save updated track
            manager.track_list[ind_track] = track
            
        # run track management 
        manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list)
        
        for track in manager.track_list:            
            print('track', track.id, 'score =', track.score)

    def associate_and_update_gnn(self, manager, meas_list, KF):
        # Associate measurements with tracks
        self.associate(manager.track_list, meas_list, KF)

        # Get optimal associations using GNN
        track_indices, meas_indices = self.GNN()

        # partially used chatgpt to generate this, googling didnt help much
        # Update associated tracks with measurements
        for ind_track, ind_meas in zip(track_indices, meas_indices):
            track = manager.track_list[ind_track]
            
            # check visibility, only update tracks in fov    
            if not meas_list[0].sensor.in_fov(track.x):
                continue
            
            # Kalman update
            print('update track', track.id, 'with', meas_list[ind_meas].sensor.name, 'measurement', ind_meas)
            KF.update(track, meas_list[ind_meas])
            
            # update score and track state 
            manager.handle_updated_track(track)
            
            # save updated track
            manager.track_list[ind_track] = track

        # Handle unassigned tracks and measurements
        self.unassigned_tracks = [i for i, t in enumerate(manager.track_list) if i not in track_indices]
        self.unassigned_meas = [i for i, m in enumerate(meas_list) if i not in meas_indices]
        self.association_matrix = np.matrix([])

        # run track management 
        manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list)
        
        for track in manager.track_list:            
            print('track', track.id, 'score =', track.score)

    # used chatgpt to generate this, googling didnt help much
    def GNN(self):
        '''Apply Global Nearest Neighbor algorithm to find the best association'''
        if self.association_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(self.association_matrix)
            return row_ind, col_ind
        return [], []