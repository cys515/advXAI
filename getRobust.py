import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import Helper
import time
import torch.utils.data as data_utils
import numpy as np
from sklearn.preprocessing import StandardScaler ,MinMaxScaler
from Helper import  checkAccuracy
import os
import multiprocessing
import dill
import warnings
import random
from attack import *
#from Plotting import *

warnings.filterwarnings("ignore", category=UserWarning)

def main(args):
	device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
	GB = ["SM", "SG", "IG"]
	# metric = ["DTW","MC","topK"]
	metric = ["MC","topK","DTW"]
	with open(args.data_dir+args.DataName+"120.dill", 'rb') as f:
		dataset = dill.load(f)
		
	Training,TrainingLabel,Testing,TestingLabel= dataset[0],dataset[1],dataset[2],dataset[3]

	args.NumTimeSteps= Training.shape[1]
	args.NumFeatures = Training.shape[2]

	train_dataRNN = data_utils.TensorDataset(torch.from_numpy(Training), torch.from_numpy(TrainingLabel))
	train_loaderRNN = data_utils.DataLoader(train_dataRNN, batch_size=1, shuffle=True)

	test_dataRNN = data_utils.TensorDataset(torch.from_numpy(Testing),torch.from_numpy(TestingLabel))
	test_loaderRNN = data_utils.DataLoader(test_dataRNN, batch_size=1, shuffle=False)

	# save np.load
	np_load_old = np.load

	# modify the default parameters of np.load
	np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

	saveModelName=args.model_dir+args.model+"/"+args.DataName
	saveModelBestName =saveModelName +"_"+str(args.step)+"_BEST.pkl"
	model = torch.load(saveModelBestName,map_location=device) 
		
	# if os.path.exists(args.Adv_Criterion_dir + args.DataName+"_"+args.model+"_"+args.XAI+"_"+args.attackMethod+".csv"):
	# 	print(args.DataName+"_"+args.model+"_"+args.XAI+"_"+args.attackMethod+" already exists.\n")
	# 	return

	if args.XAI not in GB and  args.attackMethod == "PGD":
		print("Purterbation XAI method can not use PGD attack.")
		return     

	saliency = np.load(args.Saliency_dir+args.DataName+"/"+args.model+"/"+args.XAI+".npy")

	for n in range(len(metric)):
		if os.path.exists(args.Adv_Saliency_dir+"/"+args.DataName+"/"+args.model+"/"+args.XAI+"_"+args.attackMethod+"_"+metric[n]+".npy"):
			print(args.XAI+"_"+args.attackMethod+"_"+metric[n]+" already exists.\n")
			continue
		adv_sample = np.zeros([25,Testing.shape[1],Testing.shape[2]])
		adv_saliency = np.zeros([25,Testing.shape[1],Testing.shape[2]])

		print(args.DataName+"_"+args.model+"_"+args.XAI+"_"+args.attackMethod+"_"+metric[n]+" attack\n")
		start = time.time()
		for i,  (sample, label)  in enumerate(test_loaderRNN):
			if i == 25:
				break
			print("sample:", i) 
			sample = sample.to(device)		#[num,t,f]
			module = Attack(args, sample, label, model, saliency[i,:,:], args.XAI, device, k_top=288)
			adv_sample[i,:,:], adv_saliency[i,:,:]= module.attack(args.attackMethod, metric[n], device, alpha=0.10, iters=200)
						
		end = time.time()
		avg_time = (end-start)/25

		with open(args.time_log, 'a') as fp: 
			fp.write(args.DataName+"_"+args.model+"_"+args.XAI+"_"+args.attackMethod+"_"+metric[n]+":"+str(avg_time)+'\n')
		np.save(args.Adv_dir + args.DataName+"/"+args.model+"/"+args.XAI+"_"+args.attackMethod+"_"+metric[n], adv_sample)
		np.save(args.Adv_Saliency_dir + args.DataName+"/"+args.model+"/"+args.XAI+"_"+args.attackMethod+"_"+metric[n], adv_saliency)
		print("finish")

	
	# np.savetxt(args.Adv_Criterion_dir + args.DataName+"_"+args.model+"_"+args.XAI+"_"+args.attackMethod+".csv",result, delimiter=',', header='criterion-PCC,pertub_size-PCC,criterion-MC,pertub_size-MC,criterion-topK,pertub_size-topK',fmt='%.6f')
	np.load = np_load_old


def parse_arguments(argv):
	parser = argparse.ArgumentParser()

	parser.add_argument('--DataName', type=str, default="SZtick")
	parser.add_argument('--model', type=str, default="Transformer")
	parser.add_argument('--XAI', type=str, default="SM")

	parser.add_argument('--device', type=str, default='cuda:2')
	parser.add_argument('--step', type=int, default=120)
	
	parser.add_argument('--data_dir', type=str, default="/home/cys/CYS/robust-timepoint/Datasets/")
	parser.add_argument('--model_dir', type=str, default="/home/cys/CYS/robust-timepoint/Models/")
	parser.add_argument('--Saliency_dir', type=str, default='/home/cys/CYS/robust-timepoint/Results/Saliency_Values/')
	# parser.add_argument('--Saliency_dir', type=str, default='/home/cys/CYS/robust-timepoint/Results/Smooth_Saliency/')
	parser.add_argument('--Adv_dir', type=str, default='/home/cys/CYS/robust-timepoint/Results/Adversial_Samples/')
	parser.add_argument('--Adv_Saliency_dir', type=str, default='/home/cys/CYS/robust-timepoint/Results/Adversial_Saliency/')
	# parser.add_argument('--Adv_dir', type=str, default='/home/cys/CYS/robust-timepoint/Results/Smooth_Adversial_Samples/')
	# parser.add_argument('--Adv_Saliency_dir', type=str, default='/home/cys/CYS/robust-timepoint/Results/Smooth_Adversial_Saliency/')


	parser.add_argument('--model_list', type=str, default='model_list.txt') 
	parser.add_argument('--time_log', type=str, default='timelog.txt')
	parser.add_argument('--attackMethod', type=str, default="PSO")
	return  parser.parse_args()

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))  
	