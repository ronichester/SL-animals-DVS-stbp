# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:47:50 2023

Training the SL-Animals-DVS dataset with STBP: Spatio-temporal Back Propagation

Assumes the original dataset is already sliced in 1121 samples.
(Make sure to run "slice_data.py" before training)

@author: Schechter
"""
import os
import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from datetime import datetime
from sklearn.model_selection import train_test_split
#imports from other python files
from model import SLANIMALSNet, SLANIMALSNet2
from dataset import AnimalsDvsSliced
from layers import get_args
from stbp_tools import kfold_split, train_net, test_net

#assert we are on the right working directory (good for running on containers)
PATH = os.path.dirname(os.path.realpath(__file__))
os.chdir(PATH)

#main parameters
steps = get_args()['steps']       #50 frames/time bins
dt = get_args()['dt']             #30ms per frame
batch_size = 8                    #input batch size for training
test_batch_size = 32              #input batch size for testing
epochs = 200                      #number of epochs to train
lr = 1e-3                         #learning rate (default: 1e-3 Adam | 0.5 SGD)
momentum = 0.9                    #SGD momentum (default: 0.5)
cuda = True                       #enables CUDA training (default: True)
seed = 1                          #random seed
load_model = False                #For Loading pre-trained weights on Model
data_path = '../data/'            #'/home/data/'   (YOUR DATA PATH HERE)  
model_path = './weights/'         #to save the the model weights
logs_path = './logs/'             #to save the tensorboard logs
binning_mode = 'OR'               #binning mode (OR, SUM)

#Setting the seed, use of GPU and initializing common arguments
torch.manual_seed(seed)
use_cuda = cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}  #=1
val_losses, val_accuracies = [], []       #initializing val history
test_losses, test_accuracies = [], []     #initializing test history

#creating a generator to split the data into 4 folds of train/test files
train_test_generator = kfold_split(data_path + 'filelist.txt', seed)

#----------------------------- MAIN PROGRAM ------------------------------
#main program
if __name__ == '__main__':
    
    #print header
    print('WELCOME TO STBP TRAINING!')
    print("Training params: batch size={}, epochs={}, initial LR={}, binning mode={}"
          .format(batch_size, epochs, lr, binning_mode))
    print("Simulation params: steps={}, dt={}, Vth={}, Tau={}, a1={}".format(
          steps, dt, get_args()['Vth'], get_args()['tau'], get_args()['an']))
    print('\nStarting 4-fold cross validation (train/validation/test): Please wait...\n')
    global_st_time = datetime.now()       #monitor total training time 
    
    #CROSS-VALIDATION: iterate for each fold
    for fold, (train_set, test_set) in enumerate(train_test_generator, start=1):
        
        #logging statistics with tensorboard
        writer = SummaryWriter(logs_path + 'fold{}'.format(fold))

        #divide train_set into train and validation (85-15)
        train_set, val_set = train_test_split(train_set, test_size=0.15, 
                                              random_state=seed+1)
        
        #definining train and test Datasets
        training_set = AnimalsDvsSliced(
            dataPath     = data_path,
            fileList     = train_set,
            samplingTime = dt,
            sampleLength = dt * steps ,
            fixedLength  = True, 
            binMode      = binning_mode
        )
        validation_set = AnimalsDvsSliced(
            dataPath     = data_path,
            fileList     = val_set,
            samplingTime = dt,
            sampleLength = dt * steps ,
            fixedLength  = True, 
            binMode      = binning_mode
        )
        testing_set = AnimalsDvsSliced(
            dataPath     = data_path,
            fileList     = test_set,
            samplingTime = dt,
            sampleLength = dt * steps ,
            fixedLength  = True, 
            binMode      = binning_mode
        )
        
        #definining train and test DataLoaders
        """
        If each sample on the dataset has the same length, batch_size can 
        be of any size (as long as it fits in the GPU memory). Since
        Animals-DVS dataset has different sample lengths, the only way to
        achieve training in batches is to crop all the samples to a fixed
        length size - STBP uses the first 1500 ms of all samples.
        """
        train_loader = torch.utils.data.DataLoader(dataset=training_set, 
                                                    batch_size=batch_size,
                                                    shuffle=False, **kwargs)
        val_loader = torch.utils.data.DataLoader(dataset=validation_set, 
                                                    batch_size=batch_size,
                                                    shuffle=False, **kwargs)
        test_loader = torch.utils.data.DataLoader(dataset=testing_set, 
                                                  batch_size=test_batch_size,
                                                  shuffle=False, **kwargs)
        #Defining the model
        model = SLANIMALSNet().to(device)
        # model = SLANIMALSNet2().to(device)  #under development
        
        #Defining the optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        #Training and testing along the epochs
        print("TRAINING FOLD {}:".format(fold))
        print("-----------------------------------------------")
        min_loss, max_acc = train_net(
            model, optimizer, device, train_loader, val_loader, epochs, 
            steps, writer, fold, model_path, load_model
            )
        
        #test the network
        print("TESTING FOLD {}:".format(fold))
        print("-----------------------------------------------")
        test_loss, test_acc = test_net(model, device, test_loader, writer, fold, model_path)
        
        #save this fold's losses and accuracies in history
        val_losses.append(min_loss)
        val_accuracies.append(max_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        writer.close()
        #end of fold
    
    #end of cross-validation ---------------------------------------
    global_end_time = datetime.now()     #monitor total training time
    print('\nGlobal Training Time:', global_end_time - global_st_time)
    
    #print final results
    print("\nMin Val. Loss on 4 folds:", val_losses)
    print("Min Val. Loss:     {:.2f} +- {:.2f}".format(
        np.mean(val_losses), np.std(val_losses)))

    print("\nMax Val. Accuracy on 4 folds:", val_accuracies)
    print("Max Val. Accuracy:     {:.2f}% +- {:.2f}%".format(
        np.mean(val_accuracies), np.std(val_accuracies)))

    print("\nTest Loss on 4 folds:", test_losses)
    print("Average Test Loss:     {:.2f} +- {:.2f}".format(
        np.mean(test_losses), np.std(test_losses)))

    print("\nTest Accuracy on 4 folds:", test_accuracies)
    print("Average Test Accuracy:     {:.2f}% +- {:.2f}%".format(
        np.mean(test_accuracies), np.std(test_accuracies)))
    
        