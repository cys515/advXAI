import torch
import os
import dill
from torch.autograd import Variable
import torch.utils.data as data_utils
import numpy as np
from Helper import checkAccuracy 
from Models.LSTM import LSTM
from Models.Transformer import Transformer
from Models.TCN import TCN
import torch.nn as nn


def main(args,device):
	criterion = nn.MSELoss()

	modelName = args.DataName
	saveModelName=args.model_dir+args.model+"/"+modelName+'_'+str(args.step)
	saveModelBestName =saveModelName +"_BEST.pkl"
	saveModelLastName=saveModelName+"_LAST.pkl"

	if os.path.exists(saveModelBestName) and os.path.exists(saveModelLastName):
		print("Models", args.model, modelName, "already exists.\n")
		return

	with open(args.data_dir+args.DataName+str(args.step)+".dill", 'rb') as f:
		dataset = dill.load(f)
	
	Training,TrainingLabel,Testing,TestingLabel= dataset[0],dataset[1],dataset[2],dataset[3]

	args.NumTimeSteps= Training.shape[1]
	args.NumFeatures = Training.shape[2]

	train_dataRNN = data_utils.TensorDataset(torch.from_numpy(Training), torch.from_numpy(TrainingLabel))
	train_loaderRNN = data_utils.DataLoader(train_dataRNN, batch_size=args.batch_size, shuffle=True)

	test_dataRNN = data_utils.TensorDataset(torch.from_numpy(Testing),torch.from_numpy(TestingLabel))
	test_loaderRNN = data_utils.DataLoader(test_dataRNN, batch_size=args.batch_size, shuffle=False)

	if(args.model=="LSTM"):
		net=LSTM(args.NumFeatures, args.hidden_size, args.output_size, args.rnndropout).to(device)
	elif(args.model=="Transformer"):
		net=Transformer(args.NumFeatures, args.NumTimeSteps, args.n_layers, args.heads, args.rnndropout,args.output_size,time=args.NumTimeSteps).to(device)
	elif(args.model=="TCN"):
		num_chans = [args.hidden_size] * (args.levels - 1) + [args.NumTimeSteps]
		net=TCN(args.NumFeatures,args.output_size,num_chans,args.kernel_size,args.rnndropout,time=args.NumTimeSteps).to(device)

	net.double()
	optimizerTimeAtten = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

	total_step = len(train_loaderRNN)
	Train_acc_flag=False
	Train_Acc=float('inf')
	Test_Acc=float('inf')
	BestAcc= float('inf')
	BestEpochs = 0
	patience=100

	for epoch in range(args.num_epochs):
		noImprovementflag=True
		for i, (samples, labels) in enumerate(train_loaderRNN):

			net.train()
			samples = samples.reshape(-1, args.NumTimeSteps, args.NumFeatures).to(device)
			samples = Variable(samples)
			labels = labels.to(device)
			labels = Variable(labels)

			outputs = net(samples).squeeze()
			loss = criterion(outputs, labels)

			optimizerTimeAtten.zero_grad()
			loss.backward()
			optimizerTimeAtten.step()

			if (i+1) % 3 == 0:
				Test_Acc = checkAccuracy(test_loaderRNN, net, args, device)
				Train_Acc = checkAccuracy(train_loaderRNN, net, args, device)
				if(Test_Acc<BestAcc):
					BestAcc=Test_Acc
					BestEpochs = epoch+1
					torch.save(net, saveModelBestName)
					noImprovementflag=False

				print ('{} {}-->Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Train MSE {:.2f}, Test MSE {:.2f},BestEpochs {},BestMSE {:.2f} patience {}' 
					.format(args.DataName, args.model,epoch+1, args.num_epochs, i+1, total_step, loss.item(),Train_Acc, Test_Acc,BestEpochs , BestAcc , patience))
			if(Train_Acc < 0.001):
				torch.save(net,saveModelLastName)
				Train_acc_flag=True
				break

		if(noImprovementflag):
			patience-=1
		else:
			patience=200

		if(epoch+1)%10==0:
			torch.save(net, saveModelLastName)

		if(Train_acc_flag or patience==0):
			break

		Train_Acc =checkAccuracy(train_loaderRNN , net, args, device)
		print('{} {} BestEpochs {},BestAcc {:.4f}, LastTrainAcc {:.4f},'.format(args.DataName, args.model ,BestEpochs , BestAcc , Train_Acc))

	with open(args.model_list, 'a') as fp: 
		fp.write(args.DataName+'_'+args.model+' TrainAcc:'+str(Train_Acc)+' BestAcc:'+str(BestAcc)+'\n')

