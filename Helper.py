import numpy as np
import pandas as pd
from torch.autograd import Variable
import itertools
import torch
import torch.nn as nn
from  sklearn.preprocessing import minmax_scale

def checkAccuracy(test_loader , model ,args, device, isCNN=False,returnLoss=False):
       
    model.eval()
    total_loss = 0
    total_len = 0
    criterion = nn.MSELoss()

    for  (samples, labels)  in test_loader:
        if(isCNN):
            images = samples.reshape(-1, 1,args.NumTimeSteps, args.NumFeatures).to(device)
        else:
            images = samples.reshape(-1, args.NumTimeSteps, args.NumFeatures).to(device)

        outputs = model(images).squeeze()

        labels = labels.to(device)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.shape[0]
        total_len += labels.shape[0]

    avg_loss = total_loss / total_len
    return avg_loss
