'''Explain time series and AI model with different explaining models.'''
import dill
import numpy as np
import os
import torch
from captum._utils.models.linear_model import SkLearnLasso
import torch.utils.data as data_utils
from torch.autograd import Variable
from captum.attr import (
	KernelShap,
	Lime
)

def explain(args,device):

	explain_name = args.Saliency_dir+ args.DataName + '/' + args.model + '/' + args.explain+ ".npy"
	rel_res = args.explain

	print("Explain "+explain_name+ " algorithm.")
	#load test data   
	with open(args.data_dir+args.DataName+str(args.step)+".dill" , 'rb') as f:
		dataset = dill.load(f)

	X = dataset[2][0:25]
	y = dataset[3][0:25]
	
	test_dataRNN = data_utils.TensorDataset(torch.from_numpy(X),torch.from_numpy(y))
	test_loaderRNN = data_utils.DataLoader(test_dataRNN, batch_size=1, shuffle=False)

	if os.path.exists(explain_name):
		return
		
	model_name="/home/cys/CYS/robust/Models/"+args.model+"/"+args.DataName+"_"+ str(args.step)+"_BEST.pkl"
	model = torch.load(model_name).to(device)

	results = np.zeros([X.shape[0],X.shape[1],X.shape[2]])

	for i,  (samples, labels)  in enumerate(test_loaderRNN):
		input = samples.to(device) #(10,50,50)
		labels = labels.reshape([labels.shape[0],1]).double().to(device)
		attributions = getAttribution(input,labels,model,args.explain)
		results[i] = attributions.cpu().numpy()
				
		print('Done:', i)

	np.save(explain_name, results)


def getAttribution(input,labels,pretrained_model,XAI):  #input[num,t,f] labels[num,label]
	input.requires_grad_(True)
	torch.backends.cudnn.enabled = False
	if XAI=="SM":
		attributions = SM(input, labels, pretrained_model)
	if XAI=="SG":								
		attributions = SG(input, labels, pretrained_model,50,0.1)	
	if XAI=="KS":
		KS = KernelShap(pretrained_model) 
		attributions = KS.attribute(input, n_samples=100).abs()			
	if XAI=="LIME":                 
		try:
			LIME = Lime(pretrained_model, interpretable_model=SkLearnLasso(alpha = 0.0))
			attributions = LIME.attribute(input, n_samples=100).abs()
		except e as Userwaring:
			a = 0


	a_min = torch.amin(attributions,dim=(1,2)).view(input.shape[0],1,1)
	a_max = torch.amax(attributions,dim=(1,2)).view(input.shape[0],1,1)
	epsilon = 1e-10
	attr = (attributions-a_min)/(a_max-a_min+epsilon)		
	return attr
	

def SM(input, labels, model):
	model.train()
	outputs = model(input)
	criterion = torch.nn.MSELoss()  # 使用MSE作为损失函数
	loss = criterion(outputs, labels)
	model.zero_grad()
	gradients = torch.autograd.grad(loss, input, create_graph=True)[0]
	attributions = gradients.abs()
	model.eval()
	return attributions

def SG(input, labels, model, epoch=100, alpha=0.1):
	model.train()
	sum = torch.zeros_like(input)
	criterion = torch.nn.MSELoss()  # 使用MSE作为损失函数

	for i in range(epoch):
		noise = input + torch.randn_like(input) * alpha
		noise_out = model(noise)
		noise_loss = criterion(noise_out, labels)
		model.zero_grad()
		gradients = torch.autograd.grad(noise_loss, noise, create_graph=True)[0]
		sum += gradients
	
	smooth = sum / epoch
	attributions = smooth.abs()
	model.eval()
	return attributions



