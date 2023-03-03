# -*- coding: utf-8 -*-
"""
A script to slice the SL animals DVS recordings into actual samples for 
training, and save the slices to disk. 

Make sure to run it right after downloading the dataset.

@author: Schechter
"""
#import libraries
import os
import tonic
import numpy as np
import pandas as pd

#The root data path (you can customize your path)
data_path = '../data/'

#A text file with a list of the 'raw' file names
file_list = data_path + 'filelist.txt'

#create sliced dataset directory and path
os.makedirs(data_path + "sliced_recordings", exist_ok=True)
sliced_data_path = data_path + "sliced_recordings/"

#load file names into a 1D array
files = np.loadtxt(file_list, dtype='str')  #1D array [max 59]

#check if dataset is already sliced
if len(os.listdir(sliced_data_path)) < (19 * len(files)):
    
    print('Slicing the dataset, this may take a while...')
    
    #For each of the raw recordings: slice in 19 pieces and save to disk
    for record_name in files:
        print('Processing record {}...'.format(record_name))
        
        #read the DVS file
        """
        The output of this function:
            sensor_shape: tuple, the DVS resolution (128, 128)
            events: 1D-array, the sequential events on the file
                    1 microsecond resolution
                    each event is 4D and has the shape 'xytp'
        """
        sensor_shape, events = tonic.io.read_dvs_128(data_path + 'recordings/'
                                                     + record_name + '.aedat')
        #read the tag file
        tagfile = pd.read_csv(data_path + 'tags/' + record_name + '.csv')  #df
        
        #define event boundaries for each class
        events_start = list(tagfile["startTime_ev"])
        events_end = list(tagfile["endTime_ev"])
        
        #create a list of arrays, separating the recording in 19 slices
        sliced_events = tonic.slicers.slice_events_at_indices(events, 
                                                             events_start, 
                                                             events_end)
        #save 19 separate events on disk
        for i, chosen_slice in enumerate(sliced_events):
            np.save(sliced_data_path + '{}_{}.npy'.format(
                record_name, str(i).zfill(2)), chosen_slice)
    print('Slicing completed.\n')
    
else:
    print('Dataset is already sliced.\n')

# import glob
# #create a list of the sliced dataset files
# sliced_data_list = glob.glob1(sliced_data_path, '*.npy')