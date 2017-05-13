from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

from keras.datasets import cifar10

n_train=1000
seeds = [0, 3009, 601, 2140, 15]
models = {"Linear SVM (1 vs 1)": SVC(kernel="linear"),
          "Linear SVM (1 vs all)": LinearSVC(),
          "Gaussian (rbf) (1 vs 1)": SVC(kernel="rbf"),
          "RFC (300 trees, 40 feat, 20 depth)": RandomForestClassifier(300, max_features=40, max_depth=20),
          "Logistic Regression": LogisticRegression()}

scores = pd.DataFrame({name: [None] * len(seeds) for name in models})

for i, seed in enumerate(seeds):
    print("Doing run {}".format(i+1))
    ## Data generation
    np.random.seed(seed)
    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Downsize to 1000 samples.
    classes = np.unique(y_train)
    inds_all = np.array([],dtype='int32')

    for cl in classes:
        inds = np.random.choice(np.where(np.array(y_train) == cl)[0], int(n_train/len(classes)))
        inds_all = np.r_[inds, inds_all]

    x_train = x_train[inds_all]
    y_train = y_train[inds_all]

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

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    ## Models
    for name, clf in models.items():
        print("Scoring model {}".format(name))
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        score = sum(y_pred == y_test)/len(y_pred)
        scores.loc[i, name] = score

scores.to_csv("scores.csv", index=False)
for col in scores.columns:
    print(col, scores[col].mean(), scores[col].sd())
