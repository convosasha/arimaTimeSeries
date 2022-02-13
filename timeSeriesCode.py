

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
        
        # print(self.conv1dOperator.weight)

        
    def forward(self, nSamp):
        
        # Generate vector of nSamp iid normally distributed samples
        epsVec = torch.normal(mean=0, std=self.noiseSigma, size=(1, nSamp))
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

def prepareTrainValSets(ts, minTrainSetLen, numTrainSamp, valSize):
    
    trainSetLenRange = range(minTrainSetLen, numTrainSamp - valSize + 1)

    dataSets = []
    for currTrainSetLen in trainSetLenRange:
        
        trainSet = ts[:, :, 0:currTrainSetLen].squeeze().numpy()
        valSet = ts[:, :, currTrainSetLen:(currTrainSetLen + valSize)].squeeze().numpy()

        dataSets.append([trainSet, valSet])
    
    return dataSets
    

def evaluateArima(dataSets, valSize, order):

    preds = []
    vals = []
    for [trainSet, valSet] in dataSets:
        
        model = ARIMA(trainSet, order=order)
        # model = ARIMA(trainSet, order=order, trend='t')
        # model = SARIMAX(trainSet, order=order, seasonal_order=(0,0,0,0), initialization='diffuse')
        
        fitted = model.fit()
        
        preds.extend(list(fitted.simulate(nsimulations=valSize, alpha=0.05, anchor='end')))
        
        if valSize > 1:
            vals.extend(list(valSet))
        else:
            vals.append(np.float32(valSet))

    mseScore = mean_squared_error(vals, preds)
    
    return mseScore

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

    # Set parameters to estimate
    maParamEst = Variable(torch.rand(1), requires_grad=True)
    deltaEst = Variable(torch.rand(1), requires_grad=True)
    sigmaEst = Variable(torch.rand(1), requires_grad=True)
    
    # Construct eps vector
    # epsVec = torch.zeros(numTrainSamp)
    # for kk in range(1, numTrainSamp):
    #     epsVec[kk] = tsTrain[kk] - tsTrain[kk-1] - deltaEst - maParamEst*epsVec[kk-1]

    # Set optimizer
    lr = 0.02
    optimizer = torch.optim.Adam([maParamEst, deltaEst, sigmaEst], lr=lr)
    
    nIter = 1000
    
    Lhistory = []
    maParamHistory = []
    deltaHistory = []
    sigmaHistory = []
    
    with torch.autograd.set_detect_anomaly(True):
        
        # Start training
        for iter in range(nIter):
            
            optimizer.zero_grad()
            
            Ltheta = 1
            prevEps = 0
            for kk in range(1, numTrainSamp):
                currEps = tsTrain[kk] - tsTrain[kk-1] - deltaEst - maParamEst*prevEps
                currEps = -(currEps**2)/2/sigmaEst**2
                currEps = torch.exp(currEps)
                currEps = currEps/(2*torch.pi*sigmaEst**2)**0.5
                Ltheta = Ltheta*currEps  
            
            Lhistory.append(Ltheta.data.numpy()[0])
            maParamHistory.append(maParamEst.data.numpy()[0])
            deltaHistory.append(deltaEst.data.numpy()[0])
            sigmaHistory.append(sigmaEst.data.numpy()[0])
            
            Ltheta.backward()
            
            print(f'Iteration={iter}, '
                  f'L={Ltheta.data.numpy()}, '
                  f'maParam={maParamEst.data.numpy()}, '
                  f'driftParam={deltaEst.data.numpy()}, '
                  f'sigma={sigmaEst.data.numpy()}')
            
            optimizer.step()

    # Visualize data generation stages
    corrFactor = np.max((maParamHistory, deltaHistory, sigmaHistory))/np.max(Lhistory)
    plt.figure()
    plt.plot(np.array(Lhistory)*corrFactor, label=f'ML*{corrFactor}')
    plt.plot(maParamHistory, label='ma')
    plt.plot(deltaHistory, label='drift param.')
    plt.plot(sigmaHistory, label='sigma')
    
    plt.grid(True)
    plt.title(f'Optimization history. lr={lr}, nIter={nIter}\n'
              f'ma={maParam}, driftParam={driftParam}, noiseSigma={noiseSigma}')
    plt.legend()
    plt.show()

    
    
    
    pass
    
    
    

if __name__=="__main__":
    
    os.system('cls')
    main()

