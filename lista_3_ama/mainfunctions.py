import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import log, pi, sqrt, exp

def NormZscore(data):
    mean = np.mean(data, axis = 0)
    data_int = data - mean
    sdt = np.std(data, axis = 0)
    data_norm = data_int / sdt

    return data_norm, mean, sdt

def DesnZscore(data,m,s):
    return data * s + m

def kfolds(dataset, k):
    shuf = np.random.permutation(dataset)
    n = shuf.shape[0]
    fold_size = n // k

    folds = []

    for i in range(k-1):
        folds.append(shuf[i*fold_size:(i+1)*fold_size,:])

    folds.append(shuf[(k-1)*fold_size:,:])

    return folds

def Train_Test(fold):
    traintest = []

    for i in range(len(fold)):
        test = fold[i]
        train = []
        for f in range(len(fold)):
            if i != f:
                train.extend(fold[f])

        x_train = np.array(train)[:, 0:21]
        y_train = np.array(train)[:, -1]

        x_test = np.array(test)[:, 0:21]
        y_test = np.array(test)[:, -1]

        traintest.append((x_train, y_train, x_test, y_test))

    return traintest

def ShurffleandSplit(dataset, s):
    s = round(s, 2)
    
    n = dataset.shape[0]
    shuf = np.random.permutation(dataset)
   
    r_s = round(1 - s, 2)
    perc_s = int(s * n)
    perc_r_s = int(r_s * n)

    return perc_s, perc_r_s

def covar_matrix(x):
    mean = np.mean(x, axis=0)
    x_mean = x - mean

    covar = (np.transpose(x_mean) @ x_mean) / x.shape[0]

    return covar

def confusion_matrix(cm):
    ax = sns.heatmap(cm, annot=True, cmap='Blues')

    ax.set_title('Confusion Matrix\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])

    ## Display the visualization of the Confusion Matrix.
    plt.show()

def metrics(y, y_pred):

    y = y.astype('bool')
    y_pred = y_pred.astype('bool')

    TP = sum(y & y_pred) 
    TN = sum(~y & ~y_pred)
    FP = sum(~y & y_pred)
    FN = sum(y & ~y_pred)

    precision = TP / (TP + FP)
    accuracy = (TP + TN)/(TP + FP + TN + FN)
    recall = TP /(TP+FN)
    f1 = 2*(precision * recall)/(precision + recall)
    cm = np.array([[ TP, TN ], [ FP, FN ]])

    if np.isnan(precision):
        precision = 0
    if np.isnan(accuracy):
        accuracy = 0
    if np.isnan(recall):
        recall = 0
    if np.isnan(f1):
        f1 = 0

    return precision, accuracy, recall, f1

def distance_euclidian(x1, x2):
    return sqrt(np.sum([abs(i - j) for i, j in zip(x1,x2)]))

def distance_mahalanobis(x1, x2, covar):
    inv_covar = np.linalg.inv(covar)

    return sqrt(np.sum((x1 - x2).T @ inv_covar @ (x1 - x2)))

def distance_manhattan(v, w):
    return np.sum([abs(i-j) for i, j in zip(x1,x2)])

def classes_counts(y):
    classes, counts = np.unique(y, return_counts=True)
    for i in range(len(classes)):
        print(classes[i], "=", counts[i])
    
    return classes, counts

def near_neighbors_euc(train, test, k):
    dist = np.array([])

    for r_train in train:
        dist = np.append(dist, distance_euclidian(r_train, test))
        
    idx_sorted = dist.argsort()[:k]
     
    return idx_sorted

def near_neighbors_mah(train, test, k, covar):
    dist = np.array([])

    for r_train in train:
        dist = np.append(dist, distance_mahalanobis(r_train, test, covar))
        
    idx_sorted = dist.argsort()[:k]
     
    return idx_sorted

def pred_knn_euc(train, test, k):
    neighb = near_neighbors_euc(train, test, k)
    top_neigh = [x[-1] for x in neighb]
    pred_out = max(set(top_neigh), key=top_neigh.count)

    return pred_out

def pred_knn_mah(train, test, k, covar):
    neighb = near_neighbors_mah(train, test, k, covar)
    top_neigh = [x[-1] for x in neighb]
    pred_out = max(set(top_neigh), key=top_neigh.count)

    return pred_out