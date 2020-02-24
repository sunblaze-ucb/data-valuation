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
import pdb
import copy

class Heap:
    def __init__(self,K,dist):
        self.K=K
        self.counter = -1
        self.heap = []
        self.changed = 0
        self.dist = dist
        self.heap_dist = {}

    def insert(self,a):
        dist_a = self.dist(a)
        # if a == 11:
        #     pdb.set_trace()
        if self.counter <= self.K-2:
            self.heap.append(a)
            self.heap_dist[a] = dist_a
            self.counter += 1
            self.up(self.counter)
            self.changed = 1
        else:
            if dist_a < self.dist(self.heap[0]):
                self.heap[0] = a
                self.heap_dist[a] = dist_a
                self.down(0)
                self.changed = 1
            else:
                self.changed = 0

    def up(self,index):
        if index == 0:
            return
        parent_index = (index-1)//2
        if self.dist(self.heap[index]) > self.dist(self.heap[parent_index]):
            self.heap[index],self.heap[parent_index] = self.heap[parent_index],self.heap[index]
            self.up(parent_index)
        return

    def down(self,index):
        if 2*index + 1 > self.counter:
            return
        if 2*index + 1 < self.counter:
            if self.dist(self.heap[2*index + 1]) < self.dist(self.heap[2*index + 2]):
                tar_index = 2*index + 2
            else:
                tar_index = 2*index + 1
        else:
            tar_index = 2*index + 1
        if self.dist(self.heap[index]) < self.dist(self.heap[tar_index]):
            self.heap[index],self.heap[tar_index] = self.heap[tar_index],self.heap[index]
            self.down(tar_index)
        return

def weighted_knn_class_utility_heap(y_trn,y_tst,trn_dist):
    zero_dist = np.where(trn_dist == 0)[0]
    if len(zero_dist) == 0:
        w = 1 / trn_dist
    else:
        w = np.zeros(trn_dist.shape)
        w[zero_dist] = 1
    w = w/np.sum(w)
    utility = np.sum(w*(y_trn==y_tst))
    return utility

def unweighted_knn_class_utility_heap(y_trn,y_tst,K):
    # x_tst is a single test point
    utility = np.sum((y_trn==y_tst))/K
    return utility



# multiple user

def knn_mc_approximation_adaptive_multiple_user(x_trn,y_trn,x_tst,y_tst,K,owner_data,utility,utility_class,max_iter,tol):
    n_trn = x_trn.shape[0]
    n_tst = x_tst.shape[0]
    M = len(owner_data)
    sp_approx =  np.zeros((n_tst, M))
    t_ada = np.zeros(n_tst)
    if utility_class == 'unweighted':
        for n_tst_i in range(n_tst):
            dist = lambda a: np.linalg.norm(x_trn[a,:]-x_tst[n_tst_i,:],ord=2)
            t = 0
            sp_approx_now = np.zeros(M)
            sp_approx_temp = np.zeros(M)
            while t < max_iter:
                sp_approx_prev = copy.deepcopy(sp_approx_now)
                value_now = np.zeros(M)
                perm = np.random.permutation(np.arange(M))  # data[t,:]
                heap = Heap(K=K,dist=dist)
                for k in range(M):
                    changed_vec = []
                    for data_ind in owner_data[perm[k]]:
                        heap.insert(data_ind)
                        changed_vec.append(heap.changed)
                    if np.any(changed_vec):
                        value_now[k] = utility(y_trn[heap.heap[:(heap.counter + 1)]],
                                                                         y_tst[n_tst_i], K)
                    else:
                        value_now[k] = value_now[k-1]
                # compute the marginal contribution of k-th user's data
                sp_approx_temp[perm[0]] = value_now[0]
                sp_approx_temp[perm[1:]] = value_now[1:] - value_now[0:-1]
                sp_approx_now = (sp_approx_now*t + sp_approx_temp)/(t+1)
                if np.max(np.abs(sp_approx_now-sp_approx_prev)) < tol and t >= 5:
                    break
                t += 1
                if t % 100 == 0:
                    print('%s out of %s' % (t, max_iter))
            t_ada[n_tst_i] = t
            sp_approx[n_tst_i, :] = sp_approx_now
    elif utility_class == 'weighted':
        for n_tst_i in range(n_tst):
            dist = lambda a: np.linalg.norm(x_trn[a,:]-x_tst[n_tst_i,:],ord=2)
            t = 0
            sp_approx_now = np.zeros(M)
            sp_approx_temp = np.zeros(M)
            while t < max_iter:
                sp_approx_prev = copy.deepcopy(sp_approx_now)
                value_now = np.zeros(M)
                perm = np.random.permutation(np.arange(M))  # data[t,:]
                heap = Heap(K=K,dist=dist)
                for k in range(M):
                    changed_vec = []
                    for data_ind in owner_data[perm[k]]:
                        heap.insert(data_ind)
                        changed_vec.append(heap.changed)
                    if np.any(changed_vec):
                        trn_dist = np.array([heap.heap_dist[key] for key in heap.heap[:(heap.counter + 1)]])
                        value_now[k] = weighted_knn_class_utility_heap(y_trn[heap.heap[:(heap.counter + 1)]], y_tst[n_tst_i], trn_dist)
                    else:
                        value_now[k] = value_now[k-1]
                # compute the marginal contribution of k-th user's data
                sp_approx_temp[perm[0]] = value_now[0]
                sp_approx_temp[perm[1:]] = value_now[1:] - value_now[0:-1]
                sp_approx_now = (sp_approx_now*t + sp_approx_temp)/(t+1)
                if np.max(np.abs(sp_approx_now-sp_approx_prev)) < tol:
                    break
                t += 1
                if t % 100 == 0:
                    print('%s out of %s' % (t, max_iter))
            t_ada[n_tst_i] = t
            sp_approx[n_tst_i,:] = sp_approx_now
    return sp_approx,t_ada


## single user


def knn_mc_approximation(x_trn,y_trn,x_tst,y_tst,utility_class,K,T):
    '''
    :param x_trn: training data
    :param y_trn: training label
    :param x_tst: test data
    :param y_tst: test label
    :param utility: utility function that maps a set of training instances to its utility
    :param T: the number of permutations
    :return: estimate of shapley value
    '''

    n_trn = x_trn.shape[0]
    n_tst = x_tst.shape[0]
    sp_approx_all = np.zeros((n_tst, T, n_trn))
    sp_approx =  np.zeros((n_tst, n_trn))
    if utility_class == 'unweighted':
        for n_tst_i in range(n_tst):
            dist = lambda a: np.linalg.norm(x_trn[a,:]-x_tst[n_tst_i,:],ord=2)
            for t in range(T):
                value_now = np.zeros(n_trn)
                perm = np.random.permutation(np.arange(n_trn))  # data[t,:]
                heap = Heap(K=K,dist=dist)
                for k in range(n_trn):
                    heap.insert(perm[k])
                    # if heap.changed == 1:
                    #     trn_dist = np.array([heap.heap_dist[key] for key in heap.heap[:(heap.counter+1)]])
                    #     value_now[k] = utility(y_trn[heap.heap[:(heap.counter+1)]], y_tst[n_tst_i], trn_dist)
                    # else:
                    #     value_now[k] = value_now[k-1]
                    if heap.changed == 1:
                        value_now[k] = unweighted_knn_class_utility_heap(y_trn[heap.heap[:(heap.counter + 1)]],
                                                                         y_tst[n_tst_i], K)
                    else:
                        value_now[k] = value_now[k-1]
                # compute the marginal contribution of k-th user's data
                sp_approx_all[n_tst_i, t, perm[0]] = value_now[0]
                sp_approx_all[n_tst_i, t, perm[1:]] = value_now[1:] - value_now[0:-1]
                if t % 100 == 0:
                    print('%s out of %s' % (t, T))
            sp_approx[n_tst_i,:] = np.mean(sp_approx_all[n_tst_i, :, :], axis=0)
    return sp_approx


def knn_mc_approximation_adaptive(x_trn,y_trn,x_tst,y_tst,utility_class,K,max_iter,tol):
    '''
    :param x_trn: training data
    :param y_trn: training label
    :param x_tst: test data
    :param y_tst: test label
    :param utility: utility function that maps a set of training instances to its utility
    :param T: the number of permutations
    :return: estimate of shapley value
    '''

    n_trn = x_trn.shape[0]
    n_tst = x_tst.shape[0]
    sp_approx =  np.zeros((n_tst, n_trn))
    t_ada = np.zeros(n_tst)
    if utility_class == 'unweighted':
        for n_tst_i in range(n_tst):
            dist = lambda a: np.linalg.norm(x_trn[a,:]-x_tst[n_tst_i,:],ord=2)
            t = 0
            sp_approx_now = np.zeros(n_trn)
            sp_approx_temp = np.zeros(n_trn)
            while t < max_iter:
                sp_approx_prev = copy.deepcopy(sp_approx_now)
                value_now = np.zeros(n_trn)
                perm = np.random.permutation(np.arange(n_trn))  # data[t,:]
                heap = Heap(K=K,dist=dist)
                for k in range(n_trn):
                    heap.insert(perm[k])
                    if heap.changed == 1:
                        value_now[k] = unweighted_knn_class_utility_heap(y_trn[heap.heap[:(heap.counter + 1)]],
                                                                         y_tst[n_tst_i], K)
                    else:
                        value_now[k] = value_now[k-1]
                # compute the marginal contribution of k-th user's data
                sp_approx_temp[perm[0]] = value_now[0]
                sp_approx_temp[perm[1:]] = value_now[1:] - value_now[0:-1]
                sp_approx_now = (sp_approx_now*t + sp_approx_temp)/(t+1)
                if np.max(np.abs(sp_approx_now-sp_approx_prev)) < tol:
                    break
                t += 1
                if t % 100 == 0:
                    print('%s out of %s' % (t, max_iter))
            t_ada[n_tst_i] = t
            sp_approx[n_tst_i,:] = sp_approx_now
    elif utility_class == 'weighted':
        for n_tst_i in range(n_tst):
            dist = lambda a: np.linalg.norm(x_trn[a,:]-x_tst[n_tst_i,:],ord=2)
            t = 0
            sp_approx_now = np.zeros(n_trn)
            sp_approx_temp = np.zeros(n_trn)
            while t < max_iter:
                sp_approx_prev = copy.deepcopy(sp_approx_now)
                value_now = np.zeros(n_trn)
                perm = np.random.permutation(np.arange(n_trn))  # data[t,:]
                heap = Heap(K=K,dist=dist)
                for k in range(n_trn):
                    heap.insert(perm[k])
                    if heap.changed == 1:
                        trn_dist = np.array([heap.heap_dist[key] for key in heap.heap[:(heap.counter+1)]])
                        value_now[k] = weighted_knn_class_utility_heap(y_trn[heap.heap[:(heap.counter+1)]], y_tst[n_tst_i], trn_dist)
                    else:
                        value_now[k] = value_now[k-1]
                # compute the marginal contribution of k-th user's data
                sp_approx_temp[perm[0]] = value_now[0]
                sp_approx_temp[perm[1:]] = value_now[1:] - value_now[0:-1]
                sp_approx_now = (sp_approx_now*t + sp_approx_temp)/(t+1)
                if np.max(np.abs(sp_approx_now-sp_approx_prev)) < tol:
                    break
                t += 1
                if t % 100 == 0:
                    print('%s out of %s' % (t, max_iter))
            t_ada[n_tst_i] = t
            sp_approx[n_tst_i,:] = sp_approx_now
    return sp_approx,t_ada

def knn_mc_approximation_adaptive_observe_trace_unweighted(x_trn,y_trn,x_tst,y_tst,K,max_iter,tol,until_max):
    '''
    :param x_trn: training data
    :param y_trn: training label
    :param x_tst: test data
    :param y_tst: test label
    :param utility: utility function that maps a set of training instances to its utility
    :param T: the number of permutations
    :return: estimate of shapley value
    '''

    n_trn = x_trn.shape[0]
    n_tst = x_tst.shape[0]
    sp_approx =  np.zeros((n_tst, n_trn))
    t_ada = np.zeros(n_tst)
    sp_approx_all = np.zeros((n_tst, max_iter,n_trn))
    for n_tst_i in range(n_tst):
        dist = lambda a: np.linalg.norm(x_trn[a,:]-x_tst[n_tst_i,:],ord=2)
        t = 0
        first_ind = 1
        start_time = time.time()
        sp_approx_now = np.zeros(n_trn)
        sp_approx_temp = np.zeros(n_trn)
        while t < max_iter:
            sp_approx_prev = copy.deepcopy(sp_approx_now)
            value_now = np.zeros(n_trn)
            perm = np.random.permutation(np.arange(n_trn))  # data[t,:]
            heap = Heap(K=K,dist=dist)
            for k in range(n_trn):
                heap.insert(perm[k])
                if heap.changed == 1:
                    value_now[k] = unweighted_knn_class_utility_heap(y_trn[heap.heap[:(heap.counter+1)]],
                                                                     y_tst[n_tst_i], K)
                else:
                    value_now[k] = value_now[k-1]
            # compute the marginal contribution of k-th user's data
            sp_approx_temp[perm[0]] = value_now[0]
            sp_approx_temp[perm[1:]] = value_now[1:] - value_now[0:-1]
            # print(np.max(sp_approx_temp))
            sp_approx_now = (sp_approx_now*t + sp_approx_temp)/(t+1)
            sp_approx_all[n_tst_i,t,:] = sp_approx_now
            if np.max(np.abs(sp_approx_now-sp_approx_prev)) < tol and first_ind == 1:
                t_ada[n_tst_i] = t
                first_ind = 0
                if until_max == 0:
                    break
            t += 1
            if t % 10 == 0:
                print('%s out of %s' % (t, max_iter))
                print('elapsed time is %s'%(time.time()-start_time))
                start_time = time.time()

        sp_approx[n_tst_i,:] = sp_approx_now
    return sp_approx_all,t_ada


def knn_mc_approximation_compare(x_trn,y_trn,x_tst,y_tst,utility_class,K,max_iter,tol,sp_gt):
    '''
    :param x_trn: training data
    :param y_trn: training label
    :param x_tst: test data
    :param y_tst: test label
    :param utility: utility function that maps a set of training instances to its utility
    :param T: the number of permutations
    :return: estimate of shapley value
    '''

    n_trn = x_trn.shape[0]
    n_tst = x_tst.shape[0]
    sp_approx =  np.zeros((n_tst, n_trn))
    t_ada = np.zeros(n_tst)
    if utility_class == 'unweighted':
        for n_tst_i in range(n_tst):
            dist = lambda a: np.linalg.norm(x_trn[a,:]-x_tst[n_tst_i,:],ord=2)
            t = 0
            sp_approx_now = np.zeros(n_trn)
            sp_approx_temp = np.zeros(n_trn)
            while t < max_iter:
                sp_approx_prev = copy.deepcopy(sp_approx_now)
                value_now = np.zeros(n_trn)
                perm = np.random.permutation(np.arange(n_trn))  # data[t,:]
                heap = Heap(K=K,dist=dist)
                for k in range(n_trn):
                    heap.insert(perm[k])
                    if heap.changed == 1:
                        value_now[k] = unweighted_knn_class_utility_heap(y_trn[heap.heap[:(heap.counter + 1)]],
                                                                         y_tst[n_tst_i], K)
                    else:
                        value_now[k] = value_now[k-1]
                # compute the marginal contribution of k-th user's data
                sp_approx_temp[perm[0]] = value_now[0]
                sp_approx_temp[perm[1:]] = value_now[1:] - value_now[0:-1]
                sp_approx_now = (sp_approx_now*t + sp_approx_temp)/(t+1)
                if np.max(np.abs(sp_approx_now-sp_gt)) < tol:
                    break
                t += 1
                if t % 100 == 0:
                    print('%s out of %s' % (t, max_iter))
            t_ada[n_tst_i] = t
            sp_approx[n_tst_i,:] = sp_approx_now
    return sp_approx,t_ada

