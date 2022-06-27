# kc_eight, kc_twenty = mainfunctions.ShurffleandSplit(kc, 0.8)

# x_train, x_test = x[0:kc_eight], x[-kc_twenty:]
# y_train, y_test = y[0:kc_eight], y[-kc_twenty:]

#covar_train, covar_train_inv = mainfunctions.covar_matrix(x_train, y_train)

#print(covar_train, covar_train_inv)

# import numpy as np
# import matplotlib.pyplot as plt
# import mainfunctions
# from math import log, pi, sqrt, exp
# from sklearn.model_selection import GridSearchCV, train_test_split
# from sklearn.neighbors import KNeighborsClassifier

# kc = np.genfromtxt('Aulas\Semestre 03\Machine Learning\Listas\lista_3_ama\kc2.csv', delimiter=',')
# x = kc[:,0:21]
# y = kc[:,[21]]
# x_nor, x_mean, x_sdt = mainfunctions.NormZscore(x)
# kc_nor = np.c_[x_nor, y]

# kcfolds = mainfunctions.kfolds(kc_nor, 10)

# x_train, y_train, x_test, y_test = mainfunctions.Train_Test(kcfolds)

# def dist_eucl(x1, x2):
#     return sqrt(np.sum([abs(i - j) for i, j in zip(x1,x2)]))

# def KNN1(x, y, x_t, k):
#     classes = np.unique(y)
#     results = [[dist_eucl(x[i], x_t), y[i]] for i in range(0, x.shape[0])]
#     results = sorted(results)
#     dictClasses = {}
#     for i in classes:
#         dictClasses[i] = 0
#     for i in range(0, k):
#         for row in dictClasses.keys():
#             if results[i][1] == row:
#                 dictClasses[row] += 1

#     minimus = [results[i][1] for i in range (0,k)]
    
#     contClasses = [(x, minimus.count(x)) for x in set(minimus)]

#     maximo = np.argmax(contClasses, axis=0)

#     return contClasses[maximo[1]][0]

# lista_k = [1, 5]
# hiperparamKNN = {'n_neighbors': lista_k}
# model = GridSearchCV(KNeighborsClassifier(), hiperparamKNN)
# params = model.best_params_

# print(params)

# #neighb = mainfunctions.near_neighbors(x_train, x_test, 5)
# # top_n_y = list(y[neighb])
# # pred_out = max(set(top_n_y), key=top_n_y.count)

# #print(x_test)


# #----------------------------------------------------------
# #top_neigh = [y[-1] for x in neighb]
# #pred_out = max(set(top_neigh), key=top_neigh.count)

# # print(t_k)

# # k = [1, 5]

# #kcfolds = mainfunctions.kfolds(kc_nor, 10)

# #x_train, y_train, x_test, y_test = mainfunctions.Train_Test(kcfolds)


# # dist = []
# # neigh = []
# # t_k = []

# # for i in range(len(x_train)):
# #     dist.append(mainfunctions.dist_eucl(x_train[i], x_test[i]))

# # dist.sort(key = lambda tup:tup[1])

# # for n_k in k:
# #     t_k.append(dist[0:n_k])

# # print(t_k)

# # params = model.best_params_
# # print(params)

# # print("[KNN] Hiperparâmetros escolhidos para KNN: ", model.best_params_)
    
# # print("[KNN] Testando modelo...")
# # yPredito = [KNN1(x, y, row, model.best_params_['n_neighbors']) for row in x_test]