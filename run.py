import torch
import sys
import os
import argparse
from processData import main as process_data
from train_models import main as train_models
from explain import explain
# from getRobust import main as getRobust
# import timesynth as ts
import numpy as np
import warnings

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings("ignore", category=UserWarning)
def main(args):

	#Processing data
	process_data(args)
  
	# #Train Models
	train_models(args,device)

	# #Decreasing batch size for captum  
	# args.batch_size=10  

	# #Get Saliency maps
	explain(args,device)

	# getFidelity(args,models,device)
	


def parse_arguments(argv):
	parser = argparse.ArgumentParser()
 
	parser.add_argument('--DataName', type=str, default="PRSA")
	parser.add_argument('--model', type=str, default="LSTM")
	parser.add_argument('--explain', type=str, default="LIME", help='name of explaining models [LIME | LIMNA |SHAP]')


	parser.add_argument('--plot', type=bool, default=True)
	parser.add_argument('--save', type=bool, default=True)
 
	parser.add_argument('--Frequency',type=float,default=2.0)
	parser.add_argument('--ar_param',type=float,default=0.9)
	parser.add_argument('--Order',type=int,default=10)
 
	parser.add_argument('--Graph_dir', type=str, default='/home/cys/CYS/robust-timepoint/Graphs/')
	parser.add_argument('--Saliency_Maps_graphs_dir', type=str, default='/home/cys/CYS/robust-timepoint//Graphs/Saliency_Maps/')
	parser.add_argument('--Accuracy_Drop_graphs_dir', type=str, default='/home/cys/CYS/robust-timepoint//Graphs/Accuracy_Drop/')
	
	parser.add_argument('--data_dir', type=str, default="/home/cys/CYS/robust-timepoint//Datasets/")
	parser.add_argument('--model_dir', type=str, default="/home/cys/CYS/robust-timepoint//Models/")
	parser.add_argument('--Saliency_dir', type=str, default='/home/cys/CYS/robust-timepoint//Results/Saliency_Values/')
	parser.add_argument('--Fidelity_dir', type=str, default= "/home/cys/CYS/robust-timepoint//Results/Fidelity/")
	parser.add_argument('--Adv_dir', type=str, default='/home/cys/CYS/robust-timepoint//Results/Adversial_Samples/')
	parser.add_argument('--Adv_Saliency_dir', type=str, default='/home/cys/CYS/robust-timepoint//Results/Adversial_Saliency/')
	parser.add_argument('--Adv_Criterion_dir', type=str, default='/home/cys/CYS/robust-timepoint//Results/Adversial_Criterion/')

  
	#model para
	parser.add_argument('--step', type=int, default=120, help='lag for pridiction')
	parser.add_argument('--output_size', type=int, default=1)
	parser.add_argument('--NumTimeSteps',type=int,default=51)
	parser.add_argument('--NumFeatures',type=int,default=24)
	parser.add_argument('--d_a', type=int, default=50)
	parser.add_argument('--attention_hops', type=int, default=10)      
	parser.add_argument('--n_layers', type=int, default=6)
	parser.add_argument('--heads', type=int, default=4)
	parser.add_argument('--kernel_size', type=int, default=4)
	parser.add_argument('--levels', type=int,default=3)
	parser.add_argument('--hidden_size', type=int,default=5)
	parser.add_argument('--batch_size', type=int,default=200)
	parser.add_argument('--num_epochs', type=int,default=500)
	parser.add_argument('--learning_rate', type=float,default=0.0001)
	parser.add_argument('--rnndropout', type=float,default=0.1)

	parser.add_argument('--model_list', type=str, default='model_list.txt')
	parser.add_argument('--time_log', type=str, default='timelog.txt')
	parser.add_argument('--attackMethod', type=str, default="PSO")
	return  parser.parse_args()

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))  