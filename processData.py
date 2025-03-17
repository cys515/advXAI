import os
import numpy as np
import pandas as pd
import dill
from sklearn.preprocessing import MinMaxScaler

def main(args):

	Datafile = args.data_dir+args.DataName
	data_loc = Datafile+".dill"
	if  os.path.exists(data_loc):
		return
	
	#data process
	df=pd.read_csv(Datafile+'.csv').drop(columns = "Date")
	#devide data,train:test=4:1
	test_split = round(len(df)*0.20)
	df_for_training = df[:-test_split]
	df_for_testing = df[-test_split:]
	
	#normalization
	scaler = MinMaxScaler(feature_range=(0,1))
	
	df_for_training_scaled = scaler.fit_transform(df_for_training)
	df_for_testing_scaled = scaler.transform(df_for_testing)

	#data transform to [X_step,Y]
	def createXY(dataset,n_past):
		dataX = []
		dataY = []
		for i in range(n_past, len(dataset)):
			dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
			dataY.append(dataset[i,0])
		return np.array(dataX),np.array(dataY)        

	trainX,trainY=createXY(df_for_training_scaled,args.step) #df_X=[len(train)-step,step,feature_num],df_Y=[len(train)-step]
	testX,testY=createXY(df_for_testing_scaled,args.step)

	dataset = [trainX,trainY,testX,testY]

	with open(Datafile+str(args.step)+'.dill', 'wb') as f:
		dill.dump(dataset, f)
