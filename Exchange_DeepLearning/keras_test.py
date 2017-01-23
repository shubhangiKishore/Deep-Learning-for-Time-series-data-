# using tutorial http://machinelearningmastery.com/time-series-prediction-with-deep-learning-in-python-with-keras/
# Time Series Prediction With Deep Learning in Keras
# Data Set used Exchabge Data "https://datamarket.com/data/set/22tb/exchange-rate-twi-may-1970-aug-1995#!ds=22tb&display=line"
import pandas
import matplotlib.pyplot as plt
import numpy 
from keras.models import Sequential
from keras.layers import Dense 
# Exchange Rate TWI. May 1970 ? Aug 1995

# seeding with random valu e
numpy.random.seed(7)

df = pandas.read_csv('exchange_rate.csv', usecols=[1], engine='python', skipfooter=3)
plt.plot(df)
plt.xlabel('Time')
plt.ylabel('Exchange Rate ')
#plt.show()

# multilayer perceptron regression 
# modify dataset to add a column, so that every tuples 
# has the current and next month's exchange rate 

dataset = df.values 
dataset = dataset.astype('float32')

# splitting dataset into train and test data set

train_size = int(len(dataset)*0.70)
test_size = len(dataset) - train_size
train, test = dataset[0: train_size,:], dataset[train_size:len(dataset),:]
# length of test and training dataset 
print(len(train), len(test))


def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

print create_dataset(dataset)
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY =  create_dataset (test, look_back)
model = Sequential()
model.add(Dense(8, input_dim=look_back, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=200, batch_size=2, verbose=2)
# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
# generate predictions for training
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(dataset)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
