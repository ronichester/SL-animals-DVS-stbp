import torch
import torch.nn as nn
from stbp_tools import stbp_init
from layers import get_args, tdLayer, LIFSpike

steps = get_args()['steps']

class MNISTNetMLP(nn.Module):  # MNIST MLP network on STBP paper
    def __init__(self):
        super(MNISTNetMLP, self).__init__()             #input 1x28x28
        #define the spiking layers (encapsulate non-spiking with tdLayer)
        self.fc1_s = tdLayer(nn.Linear(784, 400))         #784 = 28 * 28 * 1
        self.fc2_s = tdLayer(nn.Linear(400, 400))
        self.fc3_s = tdLayer(nn.Linear(400, 10))
        #activation function (spiking)
        self.spike = LIFSpike()
        
    def forward(self, x):
        x = x.view(x.shape[0], -1, x.shape[4])
        x = self.fc1_s(x)
        x = self.spike(x)
        x = self.fc2_s(x)
        x = self.spike(x)
        x = self.fc3_s(x)
        x = self.spike(x)
        out = torch.sum(x, dim=2) / steps  # [N, neurons, steps]
        return out
    

class MNISTNet(nn.Module):      # MNIST CNN network on STBP paper
    def __init__(self):
        super(MNISTNet, self).__init__()
        #define the spiking layers (encapsulate non-spiking with tdLayer)
        self.conv1_s = tdLayer(nn.Conv2d(1, 15, 5, 1, 2, bias=None))
        self.pool1_s = tdLayer(nn.AvgPool2d(2))
        self.conv2_s = tdLayer(nn.Conv2d(15, 40, 5, 1, 2, bias=None))
        self.pool2_s = tdLayer(nn.AvgPool2d(2))
        self.fc1_s   = tdLayer(nn.Linear(7 * 7 * 40, 300))
        self.fc2_s   = tdLayer(nn.Linear(300, 10))
        #activation function (spiking)
        self.spike   = LIFSpike()
        
    def forward(self, x):
        x = self.conv1_s(x)
        x = self.spike(x)
        x = self.pool1_s(x)
        x = self.conv2_s(x)
        x = self.spike(x)
        x = self.pool2_s(x)
        x = x.view(x.shape[0], -1, x.shape[4])
        x = self.fc1_s(x)
        x = self.spike(x)
        x = self.fc2_s(x)
        x = self.spike(x)
        out = torch.sum(x, dim=2) / steps  # [N, neurons, steps]
        return out


class NMNISTNetMLP(nn.Module):  # N-MNIST MLP network on STBP paper
    def __init__(self):
        super(NMNISTNetMLP, self).__init__()             #input  Nx2x34x34
        #define the spiking layers (encapsulate non-spiking with tdLayer)
        self.fc1_s = tdLayer(nn.Linear(2312, 400))       #2312 = 34 * 34 * 2
        self.fc2_s = tdLayer(nn.Linear(400, 400))
        self.fc3_s = tdLayer(nn.Linear(400, 10))
        #activation function (spiking)
        self.spike = LIFSpike()
    
    def forward(self, x):
        x = x.view(x.shape[0], -1, x.shape[4])  #[batch_size, 1D, time_bins]
        x = self.fc1_s(x)
        x = self.spike(x)
        x = self.fc2_s(x)
        x = self.spike(x)
        x = self.fc3_s(x)
        x = self.spike(x)
        out = torch.sum(x, dim=2) / steps  # [N, neurons, steps]
        return out    

class NMNISTNetMLP2(nn.Module):  # N-MNIST MLP network on STBP paper
    def __init__(self):
        super(NMNISTNetMLP2, self).__init__()            #input  Nx2x34x34
        #define the spiking layers (encapsulate non-spiking with tdLayer)
        self.fc1_s = tdLayer(nn.Linear(2312, 512))       #2312 = 34 * 34 * 2
        self.fc2_s = tdLayer(nn.Linear(512, 10))
        #activation function (spiking)
        self.spike = LIFSpike()
    
    def forward(self, x):
        x = x.view(x.shape[0], -1, x.shape[4])  #[batch_size, 1D, time_bins]
        x = self.fc1_s(x)
        x = self.spike(x)
        x = self.fc2_s(x)
        x = self.spike(x)
        out = torch.sum(x, dim=2) / steps  # [N, neurons, steps]
        return out    



class NMNISTNet(nn.Module):    # N-MNIST CNN network
    def __init__(self):
        super(NMNISTNet, self).__init__()                 #input  Nx 2x34x34
        #define the spiking layers (encapsulate non-spiking with tdLayer)
        self.conv1_s = tdLayer(nn.Conv2d(2, 20, 3, 1))    #output Nx20x32x32
        self.pool1_s = tdLayer(nn.AvgPool2d(2))           #output Nx20x16x16
        self.conv2_s = tdLayer(nn.Conv2d(20, 50, 3, 1))   #output Nx50x14x14
        self.pool2_s = tdLayer(nn.AvgPool2d(2) )          #output Nx50x 7x 7
        self.fc1_s = tdLayer(nn.Linear(7 * 7 * 50, 400))
        self.fc2_s = tdLayer(nn.Linear(400, 10))
        #activation function (spiking)
        self.spike = LIFSpike()
    
    def forward(self, x):
        x = self.conv1_s(x)
        x = self.spike(x)
        x = self.pool1_s(x)
        x = self.conv2_s(x)
        x = self.spike(x)
        x = self.pool2_s(x)
        x = x.view(x.shape[0], -1, x.shape[4])
        x = self.fc1_s(x)
        x = self.spike(x)
        x = self.fc2_s(x)
        x = self.spike(x)
        out = torch.sum(x, dim=2) / steps  # [N, neurons, steps]
        return out


class SLANIMALSNet(nn.Module):    # SL-Animals-DVS CNN network
    def __init__(self):
        super(SLANIMALSNet, self).__init__()        #input  Nx 2x128x128
        #define the network layers
        self.conv1 = nn.Conv2d(2, 20, 3, 1)         #output Nx20x126x126
        self.pool1 = nn.AvgPool2d(2)                #output Nx20x63x63
        self.conv2 = nn.Conv2d(20, 50, 3, 1)        #output Nx50x61x61
        self.pool2 = nn.AvgPool2d(2)                #output Nx50x30x30
        self.fc1 = nn.Linear(30 * 30 * 50, 200)     #output Nx200
        self.fc2 = nn.Linear(200, 19)               #output Nx19
        #initialize trainable weights (as in the STBP paper)
        with torch.no_grad():
            self.conv1.weight = torch.nn.Parameter(stbp_init(self.conv1.weight))
            self.conv2.weight = torch.nn.Parameter(stbp_init(self.conv2.weight))
            self.fc1.weight = torch.nn.Parameter(stbp_init(self.fc1.weight))
            self.fc2.weight = torch.nn.Parameter(stbp_init(self.fc2.weight))
        #transform into spiking layers (encapsulate non-spiking with tdLayer)
        self.conv1_s = tdLayer(self.conv1)
        self.pool1_s = tdLayer(self.pool1)
        self.conv2_s = tdLayer(self.conv2)
        self.pool2_s = tdLayer(self.pool2)
        self.fc1_s = tdLayer(self.fc1)
        self.fc2_s = tdLayer(self.fc2)
        #activation function (spiking)
        self.spike = LIFSpike()
        
    def forward(self, x):
        x = self.conv1_s(x)
        x = self.spike(x)
        x = self.pool1_s(x)
        x = self.conv2_s(x)
        x = self.spike(x)
        x = self.pool2_s(x)
        x = x.view(x.shape[0], -1, x.shape[4])      #flatten
        x = self.fc1_s(x)
        x = self.spike(x)
        x = self.fc2_s(x)
        x = self.spike(x)
        out = torch.sum(x, dim=2) / steps           # [N, neurons, steps]
        return out
