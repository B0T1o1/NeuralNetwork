import NN
import numpy as np

from sklearn.datasets import load_digits




data = load_digits()
X = data['data']
Train_test_split = 0.8
n = int(Train_test_split*len(X))
X_train = X[:n]
X_test = X[n:]
Y = data['target']

CompeletedY = []
for y in Y:
    putin = [0]*10
    putin[y] = 1
    CompeletedY.append(putin)
layers = [63,32,16,10]
network = NN.NeuralNetwork(NN.TanH(),NN.MeanSquaredError(),0.1,64,*layers)

y_train = CompeletedY[:n]
y_test = CompeletedY[n:]



network.train(X_train,y_train,10)

correct = 0
for idata in range(len(X_test)):
    result = network.forward(X_test[idata])
    if np.argmax(result) == np.argmax(y_test[idata]):
        correct += 1
    else: print(result)
print(correct/len(X_test))


