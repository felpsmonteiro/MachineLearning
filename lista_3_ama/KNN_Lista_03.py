import numpy as np
import matplotlib.pyplot as plt
import mainfunctions
from math import log, pi, sqrt, exp
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

k = [1, 5]

kc_or = np.genfromtxt('Aulas\Semestre 03\Machine Learning\Listas\lista_3_ama\kc2.csv', delimiter=',')

X = kc_or[:,0:21]
y = kc_or[:,21]

x, x_mean, x_sdt = mainfunctions.NormZscore(X)

kc = np.c_[x, y]

covar = mainfunctions.covar_matrix(x)

folds = mainfunctions.kfolds(kc, 10)

traintest = mainfunctions.Train_Test(folds)

metric_all_pred_euc = []
metric_all_pred_mah = []

for t in range(len(traintest)):
    metric_n_pred_euc = []
    metric_n_pred_mah = []
    
    for i in k:
        y_pred_euc = []
        y_pred_mah = []
    
        for r_test in traintest[t][2]:
            #Eucl
            neighb_euc = mainfunctions.near_neighbors_euc(traintest[t][0], r_test, i)
            top_n_y_euc = list(y[neighb_euc])
            pred_out_euc = max(set(top_n_y_euc), key=top_n_y_euc.count)
            y_pred_euc.append(pred_out_euc)

            #=========================================================================

            #Mahal
            neighb_mah = mainfunctions.near_neighbors_mah(traintest[t][0], r_test, i, covar)
            top_n_y_mah = list(y[neighb_mah])
            pred_out_mah = max(set(top_n_y_mah), key=top_n_y_mah.count)
            y_pred_mah.append(pred_out_mah)

        metrics_euc = mainfunctions.metrics(traintest[t][3].astype('int'), np.array(y_pred_euc).astype('int'))
        metrics_mah = mainfunctions.metrics(traintest[t][3].astype('int'), np.array(y_pred_mah).astype('int'))

        metric_n_pred_euc.append(metrics_euc)
        metric_n_pred_mah.append(metrics_mah)
    
    metric_all_pred_euc.append(metric_n_pred_euc)
    metric_all_pred_mah.append(metric_n_pred_mah)

metric_all_pred_euc = np.array(metric_all_pred_euc)
metric_all_pred_mah = np.array(metric_all_pred_mah)

meanknn_euc = np.mean(metric_all_pred_euc, axis = 0)
stdknn_euc = np.std(metric_all_pred_euc, axis = 0)

meanknn_mah = np.mean(metric_all_pred_mah, axis = 0)
stdknn_mah = np.std(metric_all_pred_mah, axis = 0)

print("\033[1m" , "\nEuclidian Distance: ", "\033[0m")

for i in range(len(k)):
    print("=" * 50)
    print("KNN k = ", k[i])
    print("\nPrecision: \t", meanknn_euc[i,0], "\nAccuracy: \t", meanknn_euc[i,1], "\nRecall: \t", meanknn_euc[i,2], \
        "\nF1Score: \t", meanknn_euc[i,3],"\nStd: \t\t", stdknn_euc[i], "\n")
    #print("Confusion Matrix: ","K = ",i, "\n", mainfunctions.confusion_matrix(cm))
    print("=" * 50, "\n")

print("\033[1m" , "\nMahalanobis Distance: ", "\033[0m")

for i in range(len(k)):
    print("=" * 50)
    print("KNN k = ", k[i])
    print("\nPrecision: \t", meanknn_mah[i,0], "\nAccuracy: \t", meanknn_mah[i,1], "\nRecall: \t", meanknn_mah[i,2], \
        "\nF1Score: \t", meanknn_mah[i,3],"\nStd: \t\t", stdknn_mah[i], "\n")
    #print("Confusion Matrix: ","K = ",i, "\n", mainfunctions.confusion_matrix(cm))
    print("=" * 50)
    



# for i in range(len(folds)):
#     test = folds[i]
#     train = []
#     for f in range(len(folds)):
#         if i != f:
#             train.extend(folds[f])

#     x_train = np.array(train)[:, 0:21]
#     y_train = np.array(train)[:, -1]

#     x_test = np.array(test)[:, 0:21]
#     y_test = np.array(test)[:, -1]

#     traintest.append((x_train, y_train, x_test, y_test))