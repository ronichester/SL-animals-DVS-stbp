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
        self.time_bins = sim_steps  #number of time bins (simulation steps)
        self.OR_mode = binning_OR   #binning mode
        
        frame_transform = transforms.Compose([
            #transform events resolution from us to ms
            transforms.Downsample(time_factor=0.001), 
            #remove offset from timestamps so that 1st event starts at t=0
            transforms.TimeAlignment(),
            #remove events outside the simulation window
            transforms.CropTime(max = sim_steps * dt),
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
        in the Tensors, and here we use 2 types of binning mode:
            -type "OR": if there is either 1 OR more spikes at a specific [x,y]
                        pixel at the same time bin, we set the value at "1.0";
            -type "SUM": set the number of spikes at each pixel for every
                        time bin as an integer number (Tonic's default mode).
        """
        if self.OR_mode:  #binning mode 'OR'
            input_spikes = torch.where((input_spikes > 0), 1.0, input_spikes)
        else:             #binning mode 'SUM'
            pass #do nothing, TonicFrames works natively in 'SUM' mode

        return (input_spikes, target)


#sliced SL-Animals-DVS dataset definition
class AnimalsDvsSliced(Dataset):
    """
    The sliced Animals DVS dataset. Much faster loading and processing!
    Make sure to run "slice_data.py" for the 1st time before using this
    dataset to slice and save the files in the correct path.
    """
    
    def __init__(self, dataPath, fileList, samplingTime, sampleLength,
                 randomCrop, binMode):
        
        self.slicedDataPath = dataPath + 'sliced_recordings/'   #string
        self.files = list_sliced_files(np.loadtxt(fileList, dtype='str')) #list [1121 files]
        self.samplingTime = samplingTime                   #30 [ms]
        self.sampleLength = sampleLength                   #1500 [ms]
        self.nTimeBins = int(sampleLength / samplingTime)  #50 bins
        self.randomCrop = randomCrop                       #boolean
        self.binMode = binMode                             #string
        #read class file
        self.classes = pd.read_csv(                        #DataFrame
            dataPath + 'SL-Animals-DVS_gestures_definitions.csv')
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        
        #load the sample file (NPY format), class name and index
        events, class_name, class_index, ss = self.get_sample(index)
        
        #prepare the target vector (one hot encoding)
        target = torch.zeros(19)            #initialize target vector
        target[class_index] = 1             #set target vector
        
        #process the events
        """
        Use this method with Tonic frames. 
        """
        #process the events
        frame_transform = transforms.Compose([
            transforms.Downsample(time_factor=0.001),  #us to ms
            transforms.TimeAlignment(),                #1st event at t=0
            transforms.ToFrame(                        #events -> frames
                sensor_size = (ss[0], ss[1], 2),
                time_window=self.samplingTime,  #in ms
                ),
            ])
        
        #transf. array of events -> frames TCWH (time_bins, 2, 128, 128)
        frames = frame_transform(events)
        
        """
        The 'frames' above has variable length for each sample.
        However, STBP needs a fixed length in order to train in batches.
        This is achieved (by default) by cropping the samples into fixed 
        crops of 1500 ms, starting from t=0. This implementation offers an
        option of using random sample crops.
        
        Information to be taken into consideration for the SL-Animals:
            - shortest sample: 880 ms. 
            - largest sample: 9466 ms
            - mean sample: 4360 +- 1189 ms stdev.
        """
        if self.randomCrop:  #choose a random crop
            actual_bins = frames.shape[0]      #actual sample length
            bin_diff = actual_bins - self.nTimeBins  #difference
            min_timebin = 0 if bin_diff <= 0 else np.random.randint(0, bin_diff)
            max_timebin = min_timebin + self.nTimeBins
            frames = frames[min_timebin:max_timebin, ...]
        else:                #crop from the beginning
            frames = frames[:self.nTimeBins, ...]
        
        #assure sample has nTimeBins (or pad with zeros)
        if frames.shape[0] < self.nTimeBins:
            padding = np.zeros((self.nTimeBins - frames.shape[0], 
                                2, ss[0], ss[1]))
            frames = np.concatenate([frames, padding], axis=0)
            
        #input spikes need to be float Tensors shaped CHWT for STBP
        frames = frames.transpose(1,3,2,0)   #TCWH -> CHWT
        input_spikes = torch.Tensor(frames)  #torch.float32

        #choice of binning mode
        """
        By default, Tonic sets the number of spikes at each pixel for every
        time bin as an integer number. Apparently, STBP uses values up to '1.0'
        in the Tensors, but we can try here 3 types of binning mode:
        -type "OR": if there is either 1 OR more spikes at a specific 
            [x,y] pixel at the same time bin, we set its value fixed at 
            "1.0 / dt" (Slayer's default mode);
        -type "SUM": set the number of spikes at each pixel for every
            time bin as an integer number (Tonic's default mode).
        -type "SUM_NORM": if there is 1 OR more spikes at a specific [x,y] 
            pixel at the same time bin, we set a value proportional to 
            the number of spikes, and so limited to the range [0.0, 1.0];
        """
        if self.binMode == 'OR' :
            #set all pixels with spikes to the value '1.0'
            input_spikes = torch.where(
                (input_spikes > 0),                 #if spike:
                1.0,                                #set pixel value to 1
                input_spikes)                       #else keep value 0
        elif self.binMode == 'SUM' :
            #pixels display the number of spikes (integer) on each time bin
            pass  #do nothing, TonicFrames works natively in 'SUM' mode
        elif self.binMode == 'SUM_NORM' :
            #set all pixels with spikes to a normalized SUM value
            input_spikes = torch.where(
                (input_spikes > 0),                 #if spike:
                input_spikes / input_spikes.max(),  #set pixel to range [0, 1.0]
                input_spikes)                       #else keep value 0
        else:
            print("Invalid binning mode; results are compromised!")
            print("(binning_mode should be only 'OR', 'SUM' or 'SUM_NORM')")
        
        return (input_spikes, target)
    
    def get_sample(self, index):
        #return the sample events, class name and class index of a sample
        assert index >= 0 and index <= 1120
   
        #the sample file name
        input_name  = self.files[index]
        
        #load the sample file (NPY format)
        events = np.load(self.slicedDataPath + input_name)
        
        #find sample class
        class_index = index % 19                           #[0-18]
        class_name =  self.classes.iloc[class_index, 1]
        
        sensor_shape = (128, 128)
        
        return events, class_name, class_index, sensor_shape
    
