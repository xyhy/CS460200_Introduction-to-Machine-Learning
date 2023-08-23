#!/usr/bin/env python
# coding: utf-8

# import packages
# Note: You cannot import any other packages!
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import random



# Global attributes
# Do not change anything here except TODO 1 
StudentID = '108062213' # TODO 1 : Fill your student ID here
input_dataroot = 'input.csv' # Please name your input csv file as 'input.csv'
output_dataroot = StudentID + '_basic_prediction.csv' # Output file will be named as '[StudentID]_basic_prediction.csv'

input_datalist =  [] # Initial datalist, saved as numpy array
output_datalist =  [] # Your prediction, should be 20 * 2 matrix and saved as numpy array
                      # The format of each row should be [Date, TSMC_Price_Prediction] 
                      # e.g. ['2021/10/15', 512]

# You can add your own global attributes here


# Read input csv to datalist
with open(input_dataroot, newline='') as csvfile:
    input_datalist = np.array(list(csv.reader(csvfile)))

# From TODO 2 to TODO 6, you can declare your own input parameters, local attributes and return parameters
    
def SplitData():
# TODO 2: Split data, 2021/10/15 ~ 2021/11/11 for testing data, and the other for training data and validation data 
    testing_data = (input_datalist[189:])
    
    validation_data = (input_datalist[168:189])
    
    data_60 = input_datalist[108:168]
    data_120 = input_datalist[48:168]
    data_168 = input_datalist[0:168]
    
    data = []
    
    data.append(testing_data)
    data.append(validation_data)
    data.append(data_60)
    data.append(data_120)
    data.append(data_168)
    
    #print(data)
    
    return data


def PreprocessData(data):
# TODO 3: Preprocess your data  e.g. split datalist to x_datalist and y_datalist
    x_datalist = []
    y_datalist = []
    ret = []
    
    
    #print(data.shape[0])
    
    for i in range(0,data.shape[0]):
        #print(data[i][1])
        x_datalist.append(int(data[i][1]))
        y_datalist.append(int(data[i][2]))
        
    ret.append(x_datalist)
    ret.append(y_datalist)
    
    return ret
        

def Regression(train):
# TODO 4: Implement regression
    x = np.array(PreprocessData(train)[0])
    y = np.array(PreprocessData(train)[1])
    
    
    x_vec = np.concatenate( (np.ones( (x.shape[0], 1) ), x[:,np.newaxis]), axis=1)
    y_vec = y[:,np.newaxis]
    
    
    w_vec = np.matmul( np.matmul( np.linalg.inv( np.matmul(x_vec.T,x_vec) ),x_vec.T ),y_vec )

    
    return w_vec


def CountLoss(model):
# TODO 5: Count loss of training and validation data
    #MSE
    #validation_data.dtype = 'float64'
    data = []
    estimate = []
    theory = []
    
    data = PreprocessData(validation_data)
    theory = data[1]
    
    beta = Regression(model)
    estimate = beta[0]+beta[1]*data[0]
    loss = (1/(estimate.size)) * np.sum((estimate-theory)**2)
    
    MAPE = 0
    for i in range(0,validation_data.shape[0]):
        MAPE += (1/estimate.size) * abs(((estimate[i]-theory[i])/theory[i]))
    
    return loss


def MakePrediction(model, target):
# TODO 6: Make prediction of testing data 

    data = []
    data = PreprocessData(target)
    w = Regression(model)
    y = w[0]+w[1]*data[0]
    
    return y



# TODO 7: Call functions of TODO 2 to TODO 6, train the model and make prediction
test = []
test = SplitData()

testing_data = test[0]
validation_data = test[1]
train_data_60 = test[2]
train_data_120 = test[3]
train_data_168 = test[4]

day60 = CountLoss(train_data_60)
day120 = CountLoss(train_data_120)
day168 = CountLoss(train_data_168)



best_model = min(day60, day120, day168)
if(best_model==day60):
    model = train_data_60
elif(best_model==day120):
    model = train_data_120
else :
    model = train_data_168
    
MakePrediction(model, testing_data)


# Write prediction to output csv
with open(output_dataroot, 'w', newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    #writer.writerow(['Date', 'TSMC Price'])
    tsmc = MakePrediction(model, testing_data)
    
    output_datalist = input_datalist
    output_datalist = np.delete(output_datalist, 2, axis=1) #date
    
    for i in range(0,20):
        output_datalist[i,0] = output_datalist[i+189,0]
        output_datalist[i,1] = round(tsmc[i])
    for j in range(20,209):
        output_datalist = np.delete(output_datalist, 20, axis=0)
    
    for row in output_datalist:
        writer.writerow(row)




