

from importlib.metadata import requires
import os

import matplotlib.pyplot as plt

import torch
import torch.nn as nn


class arima001tsModule(nn.Module):
    
    def __init__(self, maParam, driftParam):
        
        super(arima001tsModule, self).__init__()

        self.driftParam = driftParam
        
        # Define ma parameters
        self.maParam = torch.tensor(maParam)

        ma = torch.tensor([maParam, 1]) # Add zero-lag        
        ma = ma.unsqueeze(0).unsqueeze(0)
        
        # Define convolution operator
        self.conv1dOperator = nn.Conv1d(in_channels=1, out_channels=1,
                                        kernel_size=2,
                                        padding=1,
                                        bias=False)
        self.conv1dOperator.weight = nn.Parameter(ma, requires_grad=False)
        
        # print(self.conv1dOperator.weight)

        
    def forward(self, nSamp):
        
        # Generate vector of nSamp iid normally distributed samples
        epsVec = torch.normal(mean=0, std=1, size=(1, nSamp))
        epsVec = epsVec.unsqueeze(0)
        
        # Apply MA transform to generate MA vec. xt = theta1*eps_(t-1) + eps_t
        maVec = self.conv1dOperator(epsVec)[:, :, :-1]
        
        # Create ima vec. yt = x0 + ... + xt
        imaVec = torch.cumsum(maVec, dim=2)
        
        # Add drift
        driftVec = torch.cumsum(torch.ones(nSamp), dim=0)*self.driftParam
        imaDriftVec = imaVec + driftVec

          
        # Visualize data generation stages
        plt.figure()
        plt.plot(epsVec.squeeze().numpy(), label='noise')
        plt.plot(maVec.squeeze().numpy(), label='ma')
        plt.plot(imaVec.squeeze().numpy(), label='ima')
        plt.plot(imaDriftVec.squeeze().numpy(), label='ima + drift')
        
        plt.grid(True)
        plt.legend()
        plt.show(block=False)
        
        return imaDriftVec


def main():
    
    # I. Create a PyTorch module describing an ARIMA(0,1,1) time series
    maParam = 0.55
    driftParam = 0.2

    tsModule = arima001tsModule(maParam, driftParam)

    # II. Generate a random 20 sample long ARIMA(0,1,1) time series with drift
    numSamp = 20
    ts = tsModule(numSamp)
    
    






    

    
    
    
    pass
    
    
    

if __name__=="__main__":
    
    os.system('cls')
    main()



