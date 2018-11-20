'''
    Handwritten Digit Recognition
    k - Nearest Neighbors
    Merry Phan
    November 13, 2018
    Fall 2018 | CSC 3520 001
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors
import mnist
import keras

## Set relevant parameters
seed = 1
np.random.seed(seed)

X = mnist.load_images('train-images-idx3-ubyte')
T = mnist.load_labels('train-labels-idx1-ubyte')
tX = mnist.load_images('t10k-images-idx3-ubyte')
tT = mnist.load_labels('t10k-labels-idx1-ubyte')
X = X.reshape(784,-1)
tX = tX.reshape(784,-1)
X = X.T
tX = tX.T
T = T.ravel()
tT = tT.ravel()

num_samples, num_inputs = X.shape
num_classes = np.unique(T)
k = 3

## Run k-NN
classifier = neighbors.KNeighborsClassifier(k)
classifier.fit(X, T)
tX = tX[:100]
tT = tT[:100]
predict = classifier.predict(tX)
accuracy = classifier.score(tX, tT)
print("Accuracy: " + str(accuracy))

point = np.where(predict != tT)
point = np.asarray(point)
plt.imshow(tX[point[0][0],:].reshape(28,28)) # show the misclassified image
plt.title("misclassified digit")
plt.show()

# correct value for that data point predict tT[point[0][0]]
# missclassified data point predict[point[0][0]]

point = tX[point[0][0],:]
point = point.reshape(1, -1)
neighbors = classifier.kneighbors(point, return_distance=False)
n = neighbors.ravel()

plt.title("nearest neighbors")
plt.imshow(X[n[0],:].reshape(28,28))
plt.show()
plt.imshow(X[n[1],:].reshape(28,28))
plt.show()
plt.imshow(X[n[2],:].reshape(28,28))
plt.show()
