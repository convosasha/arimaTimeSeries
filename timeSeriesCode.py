

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
        epsVec = torch.normal(mean=0, std=0.5, size=(1, nSamp))
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
        # model = sarimax(trainSet, order=order, trend='c', initialization='diffuse')
        
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

    tsModule = arima001tsModule(maParam, driftParam)

    ## II. Generate a random 20 sample long ARIMA(0,1,1) time series with drift
    numSamp = 20
    ts = tsModule(numSamp)
    
    
    from statsmodels.tsa.stattools import adfuller
    dftest = adfuller(ts.squeeze().numpy(), autolag='AIC')
    print("1. ADF : ",dftest[0])
    print("2. P-Value : ", dftest[1])
    print("3. Num Of Lags : ", dftest[2])
    print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
    print("5. Critical Values :")
    for key, val in dftest[4].items():
        print("\t",key, ": ", val)
    
    
    
    ## III. Fit ARIMA model parameters using the first 14 samples of the data series
    
    # Define data sets for cross validation
    numTrainSamp = 14
    valSize = 1
    minTrainSetLen = 3

    dataSets = prepareTrainValSets(ts, minTrainSetLen, numTrainSamp, valSize)

    # Values for ARIMA parameters grid search
    pVec = [0, 1, 2, 3, 4]
    dVec = [0, 1, 2, 3]
    qVec = [0, 1, 2, 3]
    
    pVec = [0, 1, 2]
    dVec = [0, 1, 2]
    qVec = [0, 1, 2]
    
    bestOrder = (0,0,0)
    bestMse = torch.inf
    for p in pVec:
        for d in dVec:
            for q in qVec:
                
                order = (p,d,q)
                try:
                    mseScore = evaluateArima(dataSets, valSize, order)
                    
                    if mseScore < bestMse:
                        bestOrder = order
                        bestMse = mseScore

                    if order == (0,1,1):
                        reqOrderMse = mseScore
                    
                    print(f'ARIMA{order}, MSE={mseScore}')
                    
                except:
                    continue
    
    print(f"Best ARIMA order is {bestOrder} with MSE {bestMse}")
    if 'reqOrderMse' in globals(): print(f"ARIMA(0,1,1) MSE is {reqOrderMse}")

    pass
    
    
    

if __name__=="__main__":
    
    os.system('cls')
    main()
