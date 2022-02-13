

from importlib.metadata import requires
import os
import numpy as np

import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
from torch.autograd import Variable


class arima001tsModule(nn.Module):
    
    def __init__(self, maParam, driftParam, noiseSigma):
        
        super(arima001tsModule, self).__init__()

        self.driftParam = driftParam
        self.noiseSigma = noiseSigma
        
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
        
    def forward(self, nSamp):
        
        # Generate vector of nSamp iid normally distributed samples
        epsVec = torch.normal(mean=0, std=self.noiseSigma, size=(1, nSamp))
        epsVec = epsVec.unsqueeze(0)
        
        # Apply MA transform to generate MA vec. xt = theta1*eps_(t-1) + eps_t
        maVec = self.conv1dOperator(epsVec)[:, :, :-1]
        
        # Create ima vec. yt = x0 + ... + xt
        imaVec = torch.cumsum(maVec, dim=2)
        
        # Add drift
        driftVec = torch.cumsum(torch.ones(nSamp), dim=0)*self.driftParam - self.driftParam
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

def fitModelhiddenParams(nIter, numTrainSamp, tsTrain, lr, optimizer,
                         maParam, driftParam, noiseSigma,
                         maParamEst, deltaEst, sigmaEst):

    Lhistory = []
    maParamHistory = []
    deltaHistory = []
    sigmaHistory = []
    
    # Start training
    for iter in range(nIter):
        
        optimizer.zero_grad()
        
        Ltheta = 1
        prevEps = tsTrain[0]
        for kk in range(1, numTrainSamp):
            currEps = tsTrain[kk] - tsTrain[kk-1] - deltaEst - maParamEst*prevEps
            prevEps = currEps
            
            fTheta = -(currEps**2)/2/sigmaEst**2
            fTheta = torch.exp(fTheta)
            fTheta = fTheta/(2*torch.pi*sigmaEst**2)**0.5
            Ltheta = Ltheta*fTheta
        
        Lhistory.append(Ltheta.data.numpy()[0])
        maParamHistory.append(maParamEst.data.numpy()[0])
        deltaHistory.append(deltaEst.data.numpy()[0])
        sigmaHistory.append(sigmaEst.data.numpy()[0])
        
        Ltheta = -Ltheta

        Ltheta.backward()
        
        if iter % 50 == 0:
            print(f'Iteration={iter:05d}, '
                f'L={Ltheta.data.numpy()}, '
                f'maParam={maParamEst.data.numpy()}, '
                f'driftParam={deltaEst.data.numpy()}, '
                f'sigma={sigmaEst.data.numpy()}')
        
        optimizer.step()
    
    # Visualize training process
    corrFactor = np.max((maParamHistory, deltaHistory, sigmaHistory))/np.max(Lhistory)
    plt.figure()
    plt.plot(np.array(Lhistory)*corrFactor, label=f'ML*{corrFactor}')
    plt.plot(maParamHistory, label=f'ma={maParam}')
    plt.plot(deltaHistory, label=f'drift param={driftParam}')
    plt.plot(sigmaHistory, label=f'sigma={noiseSigma}')
    
    plt.grid(True)
    plt.title(f'Optimization history. lr={lr}, nIter={nIter}\n'
              f'ma={maParam}, driftParam={driftParam}, noiseSigma={noiseSigma}')
    plt.legend(loc='best')
    plt.show(block=False)

    return maParamHistory[-1], deltaHistory[-1], sigmaHistory[-1], currEps.data.numpy()[0]

def predictWithEstimatedParams(ts, tsTrain,
        numSamp, numTrainSamp,
        lr, nIter,
        maParam, driftParam, noiseSigma,
        noisesigmaEstimated, driftParamEstimated, maParamEstimated, lastEps):
    
    forecastSamples = []
    prevSamp = tsTrain[-1]
    prevEps = lastEps
    for kk in range(numSamp - numTrainSamp):
        
        currEps = torch.normal(mean=0, std=float(noisesigmaEstimated), size=(1, 1)).squeeze()
        
        currPredSamp = prevSamp + driftParamEstimated + currEps + maParamEstimated*prevEps
        forecastSamples.append(float(currPredSamp.numpy()))
        
        prevEps = currEps
        prevSamp = currPredSamp
    
    # Visualize train set, test set and predictions
    xData = range(numSamp)

    fig, ax = plt.subplots()
    ax.autoscale(enable=True, axis='y', tight=True)
    ax.plot(tsTrain, label='training')
    ax.plot(xData[numTrainSamp:], ts[numTrainSamp:], numTrainSamp, label='actual')
    ax.plot(xData[numTrainSamp:], forecastSamples, label='forecasted')
        
    ax.grid(True)
    ax.set_title(f'Train result. lr={lr}, nIter={nIter}\n'
                 f'ma={maParam}, driftParam={driftParam}, noiseSigma={noiseSigma}')
    ax.legend(loc='best')

    plt.show(block=False)


def main():
    
    ## I. Create a PyTorch module describing an ARIMA(0,1,1) time series
    maParam = 0.8
    driftParam = 0.2
    noiseSigma = 0.5

    tsModule = arima001tsModule(maParam, driftParam, noiseSigma)
    
    ## II. Generate a random 20 sample long ARIMA(0,1,1) time series with drift
    numSamp = 20
    ts = tsModule(numSamp)
    ts = ts.squeeze()
    
    ## III. Fit ARIMA model parameters using the first 14 samples of the data series
    numTrainSamp = 14
    
    # Get trainig vector
    tsTrain = ts[:numTrainSamp]
    
    # Set and initialize parameters to estimate
    maParamEst = Variable(torch.rand(1) + 0.2, requires_grad=True)
    deltaEst = Variable(torch.rand(1) + 0.2, requires_grad=True)
    sigmaEst = Variable(torch.rand(1) + 0.2, requires_grad=True)

    # Set optimizer
    lr = 0.002
    optimizer = torch.optim.Adam([maParamEst, deltaEst, sigmaEst], lr=lr)
    
    nIter = 2000
    
    maParamEstimated, driftParamEstimated, noisesigmaEstimated, lastEps = \
        fitModelhiddenParams(nIter, numTrainSamp, tsTrain, lr, optimizer,
                             maParam, driftParam, noiseSigma,
                             maParamEst, deltaEst, sigmaEst)  
    
    # Make forecast using the fitted params
    predictWithEstimatedParams(ts, tsTrain,
        numSamp, numTrainSamp,
        lr, nIter,
        maParam, driftParam, noiseSigma,
        noisesigmaEstimated, driftParamEstimated, maParamEstimated, lastEps)
    
    plt.show()
    

if __name__=="__main__":
    
    os.system('cls')
    main()
