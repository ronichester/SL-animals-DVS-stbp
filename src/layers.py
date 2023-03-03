import torch
import torch.nn as nn

# #arguments for the STBP paper: MNIST and N-MNIST datasets
# args = {
#     'steps': 15,  #stbp paper: "Time window" = 30 ms
#     'dt': 5,      #stbp paper: "Simulation time step" = 1 ms
#     'an': 0.5,    #stbp paper: "Derivative approximation parameter" = 1.0
#     'Vth': 0.3,   # V_threshold: 0.5 MNIST, 0.2 NMNIST
#     'tau': 0.3    # Leakage constant tau "Decay Factor": 0.1 MNIST, 0.2 NMNIST
# }

#arguments for the SL-Animals paper: SL-Animals-DVS dataset
args = {
    'steps': 50,  #time steps / frames (simulation window 1500ms)
    'dt': 30,     #in ms
    'an': 0.5,    #stbp paper: "Derivative approximation parameter" = 1.0
    'Vth': 0.3,   # V_threshold: 0.5 MNIST, 0.2 NMNIST
    'tau': 0.3    # Leakage constant tau "Decay Factor": 0.1 MNIST, 0.2 NMNIST
}


def get_args():
    return args

# approximate firing function
class SpikeAct(torch. autograd. Function):
    """ Defines the Spike activation function and approximates the gradient 
        according to the paper formula.
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # if input voltage (u) > Vth : output spike (1)
        return input.gt(args['Vth']).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # h1(u) is an approximate func of dg/du
        hu = (1/args['an']) * (abs(input - args['Vth']) < (args['an']/2)).float()
        return grad_input * hu

spikeAct = SpikeAct.apply

# membrane potential update
def state_update_eval(u_t_n1, o_t_n1, W_mul_o_t1_n):
    u_t1_n1 = args['tau'] * u_t_n1 * (1 - o_t_n1) + W_mul_o_t1_n
    o_t1_n1 = u_t1_n1.gt(args['Vth']).float()
    # o_t1_n1 = torch.gt(u_t1_n1 - args['Vth'], 0).float()  #old implementation!!
    
    return u_t1_n1, o_t1_n1

# membrane potential update
def state_update(u_t_n1, o_t_n1, W_mul_o_t1_n, ksi):
    u_t1_n1 = args['tau'] * u_t_n1 * (1 - o_t_n1) + W_mul_o_t1_n
    o_t1_n1 = spikeAct((u_t1_n1))
    # o_t1_n1 = spikeAct((u_t1_n1 - args['Vth']))  #old implementation!!
    
    #o_t1_n1 = F.sigmoid(1 / ksi * u_t1_n1)
    #o_t1_n1 = F.sigmoid(-(ksi. abs(). log()) * u_t1_n1)
    #o_t1_n1 = F.hardsigmoid(-(ksi.abs().log()) * u_t1_n1)
    #o_t1_n1 = F.sigmoid(u_t1_n1)
    #o_t1_n1 = F.relu(u_t1_n1)
    #o_t1_n1 = F.relu6(u_t1_n1)
    #o_t1_n1 = F.hardtanh(u_t1_n1, min_val=-1, max_val=1) + 1
    #o_t1_n1 = F.relu(u_t1_n1)
    return u_t1_n1, o_t1_n1


class tdLayer(nn.Module):
    """ Converts a common layer to the time domain. The input tensor needs to
        have an additional time dimension, which in this case is on the last 
        dimension of the data. When forwarding, a normal layer forward is 
        performed for each time step of the data in that time dimension.

    Args:
        layer (nn.Module):
            The layer needs to be converted.
        bn (nn.Module): 
            If batch-normalization is needed, the BN layer should be passed in 
            together as a parameter.
    """
    def __init__(self, layer, bn=None):
        super(tdLayer, self).__init__()
        self.layer = layer
        self.bn = bn

    def forward(self, x):
        steps = x.shape[-1]
        x_ = torch.zeros(self.layer(x[..., 0]).shape + (steps,), device=x.device)
        for step in range(steps):
            x_[..., step] = self.layer(x[..., step])

        if self.bn is not None:
            x_ = self.bn(x_)
        return x_

        
class LIFSpike(nn.Module):
    """ Generates spikes based on LIF module. It can be considered as an 
        activation function and is used similar to ReLU. The input tensor 
        needs to have an additional time dimension, which in this case is on 
        the last dimension of the data.
    """
    def __init__(self):
        super(LIFSpike, self).__init__()
        self.ksi = torch.nn.Parameter(torch.tensor([0.36]), requires_grad=True)

    def forward(self, x):
        if self. training:
            steps = x.shape[-1]
            u = torch.zeros(x.shape[:-1] , device=x.device)
            out = torch.zeros(x.shape, device=x.device)
            for step in range(steps):
                u, out[..., step] = state_update(u, 
                                                 out[..., max(step-1, 0)], 
                                                 x[..., step], 
                                                 self.ksi.to(x.device) )
            return out
        else:
            steps = x.shape[-1]
            u = torch.zeros(x.shape[:-1] , device=x.device)
            out = torch.zeros(x.shape, device=x.device)
            for step in range(steps):
                u, out[..., step] = state_update_eval(u, 
                                                      out[..., max(step-1, 0)], 
                                                      x[..., step])
            return out


class LIFVoltage(nn.Module):
    """LIF neurons that output voltage at the end.
        Generate float voltage based on LIF module.
    """
    def __init__(self):
        super(LIFVoltage, self).__init__()

    def forward(self, x):
        steps = x.shape[-1]
        u = torch.zeros(x.shape[:-1] , device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        for step in range(steps):
            u, out[..., step] = state_update(u, out[..., max(step-1, 0)], x[..., step])
        return u


class RateCoding(nn.Module):
    """Frequency encode the output.
        Rate coding of output.
    """
    def __init__(self):
        super(RateCoding, self).__init__()

    def forward(self, x):
        return torch.sum(x, dim=2) / x.shape[-1]


class Transpose(nn.Module):
    """Dimension transformation, module implementation of permute.
        Transpose the tensor. An modulize implementation of 'tensor.permute()'.
    """
    def __init__(self, *args):
        super(Transpose, self).__init__()
        self.trans_args = args

    def forward(self, x):
        return x.permute(self.trans_args)


class BroadCast(nn.Module):
    """Broadcasts legacy data as spatio-temporal domain data.
        Broadcast spatial tensor to spatial-temporal tensor.
    """
    def __init__(self):
        super(BroadCast, self).__init__()

    def forward(self, x):
        x = x.unsqueeze(len(x.shape))
        x = torch. broadcast_to(x, x. shape[:-1] + (args['steps'],))
        return x


class SNNCell(nn.Module):
    """A Wrapper for a script.
    """
    def __init__(self, snn):
        super(SNNCell, self).__init__()
        self.snn = snn

    def forward(self, x, hidden): # u[layer, batch, hidden]
        x = self.snn(x)
        return x, hidden



class tdBatchNorm(nn. BatchNorm2d):
    """Implementation of tdBN. Related paper link: https://arxiv.org/pdf/2011.05280. Specifically, in BN, it is also averaged in the time domain; and the alpha variable and Vth are introduced in the final coefficient.
        Implementation of tdBN. Link to related paper: https://arxiv.org/pdf/2011.05280. In short it is averaged over the time domain as well when doing BN.
    Args:
        num_features (int): same with nn.BatchNorm2d
        eps (float): same with nn.BatchNorm2d
        momentum (float): same with nn.BatchNorm2d
        alpha (float): an additional parameter which may change in resblock.
        affine (bool): same with nn.BatchNorm2d
        track_running_stats (bool): same with nn.BatchNorm2d
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1., affine=True, track_running_stats=True):
        super(tdBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, input):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None: # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self. num_batches_tracked)
                else: # use exponential moving average
                    exponential_average_factor = self. momentum

        # calculate running estimates
        if self. training:
            mean = input. mean([0, 2, 3, 4])
            # use biased var in train
            var = input.var([0, 2, 3, 4], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self. running_mean
            var = self. running_var

        input = self. alpha * args['Vth'] * (input - mean[None, :, None, None, None]) / (torch. sqrt(var[None, :, None, None, None] + self. eps))
        if self.affine:
            input = input * self.weight[None, :, None, None, None] + self.bias[None, :, None, None, None]

        return input