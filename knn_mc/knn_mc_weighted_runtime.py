import numpy as np
import scipy.special
from itertools import combinations
from compute_knn_shapley import get_true_KNN,compute_single_unweighted_knn_class_shapley
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
import time
import math
import operator
from functools import reduce
from compute_single_weighted import compute_weighted_knn
from sklearn import preprocessing
from compute_single_weighted import weighted_knn_class_utility
from compute_bound import benett_bound
from knn_mc_heap import knn_mc_approximation,weighted_knn_class_utility_heap,knn_mc_approximation_adaptive

np.random.seed(0)
# approximation algorithm to compute the weighted KNN value
data = np.load('./data/dog-fish_inception_features_new_train.npz')
x_trn_all = data['inception_features_val']
y_trn_all = data['labels']
data.close()


data = np.load('./data/dog-fish_inception_features_new_test.npz')
x_tst_all = data['inception_features_val']
y_tst_all = data['labels']
data.close()

K = 3
epsilon = 0.01
delta = 0.01
N_vec = np.concatenate((np.array([2]),np.arange(10,100,10)))
runtime = np.zeros(len(N_vec))

#
# for Ni in range(len(N_vec)):
#     N = N_vec[Ni]
#     pos_ind = np.where(y_trn_all == 0)[0]
#     neg_ind = np.where(y_trn_all == 1)[0]
#     n_trn = N  # 500
#
#     rand_ind_pos = np.random.choice(pos_ind, int(n_trn / 2))
#     rand_ind_neg = np.random.choice(neg_ind, int(n_trn / 2))
#     rand_ind = np.concatenate((rand_ind_neg, rand_ind_pos))
#
#     x_trn_raw = x_trn_all[rand_ind, :]
#     y_trn = y_trn_all[rand_ind]
#
#     pos_ind_tst = np.where(y_tst_all == 0)[0]
#     neg_ind_tst = np.where(y_tst_all == 1)[0]
#
#     n_tst = 50  # 100
#     rand_ind_pos_tst = np.random.choice(pos_ind_tst, int(n_tst / 2))
#     rand_ind_neg_tst = np.random.choice(neg_ind_tst, int(n_tst / 2))
#     rand_ind_tst = np.concatenate((rand_ind_neg_tst, rand_ind_pos_tst))
#     x_tst_raw = x_tst_all[rand_ind_tst, :]
#     y_tst = y_tst_all[rand_ind_tst]
#
#     scaler = preprocessing.StandardScaler().fit(x_trn_raw)
#     x_trn = scaler.transform(x_trn_raw)
#     x_tst = scaler.transform(x_tst_raw)
#
#     N_tst = x_tst.shape[0]
#     N_trn = x_trn.shape[0]
#
#     T_bound = benett_bound(n=n_trn,K=K,r=1/K,epsilon=epsilon,delta=delta)
#     T = int(np.round(T_bound))
#     print('T is %s'%T)
#
#     result = {}
#     # sp_approx_all = knn_mc_approximation(x_trn,y_trn,x_tst[:1,:],y_tst[:1],weighted_knn_class_utility_heap,K,T)
#     start_time = time.time()
#     sp_approx_ada,t_ada = knn_mc_approximation_adaptive(x_trn,y_trn,x_tst[:1,:],y_tst[:1],weighted_knn_class_utility_heap,K,T,epsilon/100)
#     runtime[Ni] = time.time() - start_time
#
#     result['sp_approx'] = sp_approx_ada
#     result['ada'] = t_ada
#     result['bennett'] = T
#     np.save('./result/weighted_knn_approx_result_'+str(n_trn)+'_heap.npy', result)
#     np.save('./result/weighted_knn_approx_runtime.npy',runtime)



## runtime as a funciton of K

i_tst = 0

pos_ind = np.where(y_trn_all == 0)[0]
neg_ind = np.where(y_trn_all == 1)[0]
n_trn = 100

rand_ind_pos = np.random.choice(pos_ind, int(n_trn / 2))
rand_ind_neg = np.random.choice(neg_ind, int(n_trn / 2))
rand_ind = np.concatenate((rand_ind_neg, rand_ind_pos))

x_trn_raw = x_trn_all[rand_ind, :]
y_trn = y_trn_all[rand_ind]

pos_ind_tst = np.where(y_tst_all == 0)[0]
neg_ind_tst = np.where(y_tst_all == 1)[0]

n_tst = 50  # 100
rand_ind_pos_tst = np.random.choice(pos_ind_tst, int(n_tst / 2))
rand_ind_neg_tst = np.random.choice(neg_ind_tst, int(n_tst / 2))
rand_ind_tst = np.concatenate((rand_ind_neg_tst, rand_ind_pos_tst))
x_tst_raw = x_tst_all[rand_ind_tst, :]
y_tst = y_tst_all[rand_ind_tst]

scaler = preprocessing.StandardScaler().fit(x_trn_raw)
x_trn = scaler.transform(x_trn_raw)
x_tst = scaler.transform(x_tst_raw)
K_vec = np.arange(2,10)
runtime_K = np.zeros(len(K_vec))
for ki in range(len(K_vec)):
    K = K_vec[ki]
    T_bound = benett_bound(n=n_trn, K=K, r=1 / K, epsilon=epsilon, delta=delta)
    T = int(np.round(T_bound))
    print('T is %s' % T)
    result = {}
    start_time = time.time()
    sp_approx_ada, t_ada = knn_mc_approximation_adaptive(x_trn, y_trn, x_tst[:1, :], y_tst[:1],
                                                         weighted_knn_class_utility_heap, K, T, epsilon / 100)
    runtime_K[ki] = time.time() - start_time
    result['sp_approx'] = sp_approx_ada
    result['ada'] = t_ada
    result['bennett'] = T
    print('---')
    print('%s out of %s is finished!' % (ki, len(K_vec)))
    print('total elapsed time is %s ' % runtime_K[ki])
    np.save('./result/weighted_knn_approx_result_K_' + str(K) + '_heap.npy', result)
    np.save('./result/weighted_knn_approx_runtime_K_N_100.npy', runtime_K)

# examine the results



# convergence
# n_trn = 30
# sp_approx_all = np.load('./result/weighted_knn_approx_sp_'+str(n_trn)+'_heap.npy')
# sp_approx_test = sp_approx_all[0,:,:]
# n_trn = sp_approx_test.shape[1]
# T_bound = benett_bound(n=n_trn,K=K,r=1/K,epsilon=epsilon,delta=delta)
# T = int(np.round(T_bound))
# sp_value = np.zeros((T,n_trn))
# err_value = np.zeros(T)
# sp_gt =np.load('./result/weighted_N30_K_1.npy')
# for i in range(1,T):
#     sp_value[i,:] = np.mean(sp_approx_test[:i,:],axis=0)
#     err_value[i] = np.linalg.norm(sp_gt-sp_value[i,:],ord=np.inf)
#
# plt.plot(range(T),sp_value)
# plt.show()
#
# plt.semilogy(err_value)
# plt.show()
#
# sp_gt =np.load('./result/weighted_N30_K_1.npy')
# sp_approx_final = sp_value[-1,:]
# error = np.linalg.norm(sp_gt-sp_approx_final,ord=np.inf)
# print(error)
#
# plt.plot(sp_gt,label='ground truth')
# plt.plot(sp_approx_final,label='approximation')
# plt.legend()
# plt.show()
#
# plt.plot(sp_gt,label='ground truth')
# plt.plot(sp_approx_ada.T,label='approximation-ada')
# plt.legend()
# plt.show()
# error = np.linalg.norm(sp_gt-sp_approx_ada[0,:],ord=np.inf)
# print(error)



# dist = np.zeros(n_trn)
#
# for i in range(n_trn):
#     dist[i] = np.linalg.norm(x_trn[i,:] -x_tst[0,:],ord = 2)
# sort_ind = np.argsort(dist)
# print(sp_gt[sort_ind[-1]])
# print(sp_approx_final[sort_ind[-1]])