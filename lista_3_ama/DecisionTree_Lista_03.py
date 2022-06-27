import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import mainfunctions
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics, model_selection
from math import log, pi, sqrt, exp

kc_or = np.genfromtxt('Aulas\Semestre 03\Machine Learning\Listas\lista_3_ama\kc2.csv', delimiter=',')
X = kc_or[:,0:21]
y = kc_or[:,21]
x, x_mean, x_sdt = mainfunctions.NormZscore(X)
kc = np.c_[x, y]

folds = mainfunctions.kfolds(kc, 10)

traintest = mainfunctions.Train_Test(folds)
# x_train, y_train, x_test, y_test


criteria = ['entropy', 'gini']
#metrics_all = []

for c in range(len(criteria)):
    metrics_criteria = []

    for t in range(len(folds)):
        clf = DecisionTreeClassifier(criterion = criteria[c])
        clf = clf.fit(traintest[t][0],traintest[t][1])

        y_pred = clf.predict(traintest[t][2])
        metrics = mainfunctions.metrics(traintest[t][3].astype('int'), np.array(y_pred).astype('int'))
        
        metrics_criteria.append(metrics)

    mean = np.mean(metrics_criteria, axis = 0)
    std = np.std(metrics_criteria, axis = 0)

    print("\033[1m" , "\nCriteria: ", criteria[c],"\033[0m")
    print("=" * 50)
    print("\nPrecision: \t", mean[0], "\nAccuracy: \t", mean[1], "\nRecall: \t", mean[2], \
        "\nF1Score: \t", mean[3],"\nStd: \t\t", std, "\n")
    #print("Confusion Matrix: ","K = ",i, "\n", mainfunctions.confusion_matrix(cm))
    print("=" * 50, "\n")
