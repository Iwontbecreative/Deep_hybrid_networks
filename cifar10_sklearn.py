from sklearn.svm import SVC, LinearSVC
import numpy as np

from keras.datasets import cifar10

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Downsize to 1000 samples.
idx = np.random.choice(np.arange(len(x_train)), 1000, replace=False)
x_train = x_train[idx]
y_train = y_train[idx]

# Flatten for sklearn
y_train = y_train.flatten()
y_test = y_test.flatten()
n_labels = len(np.unique(y_train))

n_train = x_train.shape[0]
n_test = x_test.shape[0]
height = x_train.shape[1]
width = x_train.shape[2]
n_channels = x_train.shape[3]

x_train = x_train.reshape((n_train,height*width*n_channels))
x_test = x_test.reshape((n_test,height*width*n_channels))

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Model

# Linear SVM
clf = SVC(kernel='linear')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("Linear SVM accuracy (1 vs 1):", sum(y_pred == y_test)/len(y_pred))

clf = LinearSVC()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("Linear SVM accuracy (1 vs all):", sum(y_pred == y_test)/len(y_pred))
