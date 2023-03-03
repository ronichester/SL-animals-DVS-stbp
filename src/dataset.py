# import numpy as np
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tonic.datasets import NMNIST
import tonic.transforms as transforms
from stbp_tools import list_sliced_files

    
class MyNMNIST(Dataset):
    """
    A customized N-MNIST implementation, for fast and light STBP training.
    """
    def __init__(self, train, sim_steps, dt, path, binning_OR=True):
        """
        Arguments:
            train: bool; 
                True for training set | False for testing set
            sim_steps: int; 
                Number of simulation steps (or time bins)
            dt: int; 
                Sampling time in ms
            path: str; 
                Where to store the data
            binning_OR: bool; 
                Binning mode for setting the spike values. If 'True' (default)
                considers only pixel active or inactive (1 or 0) for that
                time bin; else (False) considers a value proportional to the
                number of spikes at every pixel (binning mode 'SUM').
            
        """
        super(MyNMNIST, self).__init__()
        self.train = train          #if True use train set otherwise test set
        self.time_bins = sim_steps  #number of time bins (simulation steps)
        self.dt = dt                #sampling time in ms
        self.path = path            #data path
        self.OR_mode = binning_OR   #binning mode
        self.win = sim_steps * dt   #simulation window in ms (length)
        
        frame_transform = transforms.Compose([
            #transform events resolution from us to ms
            transforms.Downsample(time_factor=0.001), 
            #remove offset from timestamps so that 1st event starts at t=0
            transforms.TimeAlignment(),
            #remove events outside the simulation window
            transforms.CropTime(max=self.win),
            #transform event into frames using time bins
            transforms.ToFrame(sensor_size = NMNIST.sensor_size,
                               time_window = dt)
            ])
        
        self.dataset = NMNIST(save_to=path, train=train, 
                              transform=frame_transform)
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Gets a dataset item by it's index.
        """
        #read the data
        frames = self.dataset[idx][0]       #array TCWH (time_bins, 2, 34, 34)
        class_index = self.dataset[idx][1]  #the label (int)
        
        #prepare the target vector (one hot encoding)
        target = torch.zeros(10)            #initialize target vector
        target[class_index] = 1             #set target vector
        
        #input spikes need to be of type torch.FloatTensor reshaped to CHWT
        T, C, W, H = frames.shape  #(frames.dtype = np.int32)
        input_spikes = torch.Tensor(frames).reshape(C, H, W, T) #torch.float32
        
        #if sample has less than nTimeBins, fill missing bins with zeros
        if input_spikes.shape[-1] < self.time_bins:
            padding = torch.zeros((NMNIST.sensor_size[-1],   #2
                                   NMNIST.sensor_size[0],   #34
                                   NMNIST.sensor_size[1],   #34
                                   self.time_bins - input_spikes.shape[-1]))
            input_spikes = torch.cat([input_spikes, padding], dim=-1)
        
        #choice of binning mode
        """
        By default, Tonic sets the number of spikes at each pixel for every
        time bin as an integer number. Apparently, STBP uses values up to '1.0'
        in the Tensors, and therefore, we can use 2 types of binning mode.
            -type "OR": if there is either 1 OR more spikes at a specific [x,y]
                        pixel at the same time bin, we set the value at "1.0";
            -type "SUM": if there is 1 OR more spikes at a specific [x,y] pixel
                         at the same time bin, we set a value proportional to 
                         the number of spikes, and limited to '1.0'.
        """
        if self.OR_mode:  #binning mode 'OR'
            input_spikes = torch.where((input_spikes > 0), 1.0, input_spikes)
        else:             #binning mode 'SUM'
            input_spikes = torch.where(
                (input_spikes > 0),  #if there is a spike in this pixel:
                input_spikes/input_spikes.max(),  #change value to this
                input_spikes)                     #else keep value at 0

        return (input_spikes, target)


#sliced SL-Animals-DVS dataset definition
class AnimalsDvsSliced(Dataset):
    """
    The sliced Animals DVS dataset. Much faster loading and processing!
    Make sure to run "slice_data.py" for the 1st time before using this
    dataset to slice and save the files in the correct path.
    """
    
    def __init__(self, dataPath, fileList, samplingTime, sampleLength,
                 fixedLength, binMode):
        
        self.path = dataPath                               #string
        self.slicedDataPath = dataPath + 'sliced_recordings/'   #string
        self.files = list_sliced_files(fileList)           #list [1121 files]
        self.samplingTime = samplingTime                   #30 [ms]
        self.sampleLength = sampleLength                   #1500 [ms]
        self.nTimeBins = int(sampleLength / samplingTime)  #50 bins 
        self.fixedLength = fixedLength                     #boolean
        self.binMode = binMode                             #string
        #read class file
        self.classes = pd.read_csv(                        #DataFrame
            self.path + 'SL-Animals-DVS_gestures_definitions.csv')
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        assert index >= 0 and index <= 1120
   
        #the sample file name
        input_name  = self.files[index]
        
        #load the sample file (NPY format)
        events = np.load(self.slicedDataPath + input_name)
        
        #find sample class
        class_index = index % 19                           #[0-18]
        # class_name =  self.classes.iloc[class_index, 1]
        
        #prepare the target vector (one hot encoding)
        target = torch.zeros(19)            #initialize target vector
        target[class_index] = 1             #set target vector
        
        #process the events
        """
        Use this method with Tonic frames (fixed OR variable time_bins).
        """
        #if using fixed sampleLength/time_bins, crop relevant events
        if self.fixedLength:
            frame_transform = transforms.Compose([
                transforms.Downsample(time_factor=0.001),    #us to ms
                transforms.TimeAlignment(),         #1st event at t=0
                transforms.CropTime(max=self.sampleLength),  #crop events
                transforms.ToFrame(                 #events -> frames
                    sensor_size = (128, 128, 2),
                    time_window=self.samplingTime,  #in ms
                    )
                ])
        else:  #variable length
            frame_transform = transforms.Compose([
                transforms.Downsample(time_factor=0.001),  #us to ms
                transforms.TimeAlignment(),                #1st event at t=0
                transforms.ToFrame(                        #events -> frames
                    sensor_size = (128, 128, 2),
                    time_window=self.samplingTime,  #in ms
                    )
                ])
        
        #transf. array of events -> frames TCWH (time_bins, 2, 128, 128)
        frames = frame_transform(events)
        
        #input spikes need to be float Tensors reshaped to CHWT for STBP
        T, C, W, H = frames.shape
        input_spikes = torch.Tensor(frames).reshape(C, H, W, T) #torch.float32
        
        #if fixedLength, assure sample has nTimeBins (or pad with zeros)
        if self.fixedLength:
            if input_spikes.shape[-1] < self.nTimeBins:
                padding = torch.zeros(
                    (2, 128, 128, self.nTimeBins - input_spikes.shape[-1]))  
                input_spikes = torch.cat([input_spikes, padding], dim=-1)

        #choice of binning mode
        """
        By default, Tonic sets the number of spikes at each pixel for every
        time bin as an integer number. Apparently, STBP uses values up to '1.0'
        in the Tensors, and therefore, we can use 2 types of binning mode.
            -type "OR": if there is either 1 OR more spikes at a specific [x,y]
                        pixel at the same time bin, we set the value at "1.0";
            -type "SUM": if there is 1 OR more spikes at a specific [x,y] pixel
                         at the same time bin, we set a value proportional to 
                         the number of spikes, and limited to '1.0'.
        """
        if self.binMode == 'OR' :
            #set all pixels with spikes to the value '1.0'
            input_spikes = torch.where(
                (input_spikes > 0),   #if spike:
                1.0,                                #set pixel value
                input_spikes)                       #else keep value 0
        elif self.binMode == 'SUM' :
            pass  #do nothing, TonicFrames work natively in 'SUM' mode
        else:
            print("Invalid binning mode; results are compromised!")
            print("(binning_mode should be only 'OR' or 'SUM')")
        
        return (input_spikes, target)
    
