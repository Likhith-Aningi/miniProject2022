import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
#
import PySimpleGUI as sg
#
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
#from tensorflow.keras.layers import Dense,Dropout,LSTM

#load the data.

print(os.getcwd())
dic={"AMAZON":"AMZN","FLIPKART":"FPKT","RELIANCE":"RELIANCE","META":"META","MAXLINEAR":"MXL"}
#
sg.theme('DarkAmber')
layout=[[sg.Text('Select respective company button to predict it shares')],
		[],
		[sg.Button('Amazon',key='-A-'),sg.Button('Netflix',key='-N-'),sg.Button('Apple',key='-AA-'),sg.Button('Meta',key='-Me-'),sg.Button('Maxlinear',key='-Mx-')],
		[sg.Text('',key='-result-')],
		[sg.Button('Exit',button_color=('white','black'),font=('Helvetica', 14))]

		]
window=sg.Window('miniprojet',layout,icon='resources\\Logonew.ico')
#dic={"AMAZON":"AMZN","FLIPKART":"FPKT","RELIANCE":"RELIANCE","META":"META","MAXLINEAR":"MXL"}
while True:
	event,values=window.read()
	if event is None or event=='Exit':
		break
	if event=='-A-':
		print('amazon')
		company="AMZN"
		data=pd.read_csv('resources\\Amzn.csv')
		model=load_model('resources\\Amznmodel')
		break
	if event=='-AA-':
		print('apple')
		company="AAPL"
		data=pd.read_csv('resources\\Apl.csv')
		model=load_model('resources\\Applemodel')
		break
	if event=='-N-':
		print('Netflix')
		company='NFLX'
		data=pd.read_csv('resources\\Ntflix.csv')
		model=load_model('resources\\Netflixmodel')
		break
	if event=='-Me-':
		print('Meta')
		company='META'
		data=pd.read_csv('resources/Meta.csv')
		model=load_model('resources/Metamodel')
		break
	if event=='-Mx-':
		print('Maxilinear')
		company='MXL'
		data=pd.read_csv('resources/Mxl.csv')
		model=load_model('resources/Mxlmodel')
		break			

#event,values=window.read()
#
#print(dic)
print("choose the company's ticker from the above dictionary ")
#company=input("enter ticker symbol  :")
      

start = dt.datetime(2013,1,1)
end   = dt.datetime(2022,1,1)
#data = web.DataReader(company,'yahoo',start,end)
#data=pd.read_csv('C:\\Users\\Likhith\\Desktop\\e3s2\\miniproject\\ex.csv')


#prepare the data for neural network.

scaler= MinMaxScaler(feature_range=(0,1))
scaled_data= scaler.fit_transform(data['Close'].values.reshape(-1,1))


prediction_days = 60

x_train=[]
y_train=[]

for x in range(prediction_days,len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x,0])
    y_train.append(scaled_data[x,0])

x_train,y_train= np.array(x_train),np.array(y_train)

#x_train is going to work with neural network. so we have to reshaped it 

x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))


#build the model 

#build the model
#model=load_model('Mxlmodel')
'''
model=Sequential()

#we always one lstm layers and one dropout layers and one lstm layer and one dropout layer 
#in the end we add dense layer,this in one unit 
#that one unit is going to predict the stock price 

#if we add more layers and more units we have to train the model for longer time 
#if we use more layers it may overfit of sophistication

model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1)) #prediction of the next closing value

model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train,epochs=100,batch_size=32)

'''
#Testing the model accuracy on existing data

#load test data
#this data had to be the data that the model has not seen before 

test_start = dt.datetime(2022,1,1)

test_end = dt.datetime.now() 

test_data = web.DataReader(company,'yahoo',test_start,test_end)

#we need to scale the prices and we need to concatenate a full data set of the data the we predict 

actual_prices= test_data['Close'].values

 
#total_data set which combines  training dataset and testing dataset 


total_dataset = pd.concat((data['Close'],test_data['Close']),axis=0)

#we need to give the input to the model so it can predict the next price

model_inputs= total_dataset[len(total_dataset)- len(test_data)- prediction_days:].values
model_inputs= model_inputs.reshape(-1,1)
model_inputs= scaler.transform(model_inputs)


#make predictions on test data

x_test =[]

for x in range(prediction_days,len(model_inputs)):
	x_test.append(model_inputs[x-prediction_days:x,0])

x_test  = np.array(x_test)
x_test =  np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))


predicted_prices = model.predict(x_test)

#predicted prices are scaled,so we need to reverse scaled them. so we can get back to the actucal prices.

predicted_prices = scaler.inverse_transform(predicted_prices)


#plot the test predictions.


plt.plot(actual_prices,color='black',label=f"Actual {company} Price")
plt.plot(predicted_prices,color='green',label=f"Prediction {company} Price")
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f'{company} Share price')
plt.legend()
plt.show()


#Predict the next day

real_data=[model_inputs[len (model_inputs)+1 - prediction_days : len(model_inputs+1),0]]
real_data=np.array(real_data)
real_data=np.reshape(real_data,(real_data.shape[0],real_data.shape[1],1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction : {prediction}")
window['-result-'].update(f'Predicted stock Price of {company}: {" USD "+str(prediction[0])[1:7]}')
while True:
	event,values=window.read()
	
	if event=='Exit'or event==sg.WIN_CLOSED:
		break
#saving the model
#model.save('lstmodel')



window.close()
