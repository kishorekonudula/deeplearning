# Neural network with keras (taken from http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)
from keras.models import Sequential
from keras.layers import Dense
from keras import backend
import numpy
# fit random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("C://Users//kkonudul//Downloads//training//deeplearning//pima-indians-diabetes.csv", delimiter = ",")
X = dataset[:,0:8]
Y = dataset[:,8]

# create model
model = Sequential()
model.add(Dense(12, input_dim = 8, activation = backend.relu))
model.add(Dense(8, activation = backend.relu))
model.add(Dense(1, activation = backend.sigmoid))

# compile model
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# fit the model
model.fit(X, Y, epochs = 150, batch_size = 10)

# evaluate the model
scores = model.evaluate(X, Y)
print("{} : {}".format(model.metrics_names[1], scores[1]*100))

# calculate predictions
predictions = model.predict(X)

# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)