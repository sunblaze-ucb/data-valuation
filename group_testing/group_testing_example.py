import numpy as np
import pickle
import time
from utility import estimate_testing_loss_weight
from sklearn import datasets
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt
from cvxpy import *


seed = 0
np.random.seed(seed)


## import data and take two classes
iris = datasets.load_iris()
x = iris.data
y = iris.target
x = x[np.where(y!=2)[0],:]
x_mean = np.linalg.norm(x,ord=2,axis=1)
x = x/np.max(x_mean)
y = y[np.where(y!=2)[0]]
y = 2*y-1 # change the label (0,1) to (-1,1)
permute_ind = np.random.permutation(x.shape[0])
x = x[permute_ind,:]
y = y[permute_ind]
n_all = x.shape[0]
n_trn = 40
n_tst = n_trn
x_trn,y_trn = x[:n_trn,:],y[:n_trn]
x_tst,y_tst = x[n_trn:n_trn+n_tst,:],y[n_trn:n_trn+n_tst]



## logistic regression parameters
regularizer = 0.01
max_loss = np.log(2) # range of the uility
epsilon = 1/(n_trn)
delta = 0.05


## group testing parameters
r = max_loss # range of utility
N = n_trn
Z = 2*np.sum([1/i for i in range(1,N)])
T = int(np.ceil(2*(r**2)*(Z**2)*np.log(N*(N-1)/delta)/(epsilon**2)))

def h(x):
    y = (1+x)*np.log(1+x) - x
    return y

q = [1 / Z * (1 / k + 1 / (N - k)) for k in range(1, N)]
q_tot = q[0]
for j in range(1, N-1):
    k = j + 1
    q_tot += q[j] * (1 + 2 * k * (k - N) / (N*(N-1)))
T_new = 4/(1-q_tot**2)/h(2*epsilon/Z/r/(1-q_tot**2))*np.log(N*(N-1)/(2*delta))
T_new = int(np.ceil(T_new))

## parallelise trials
def paralell_sample_group_testing(t,x_trn,y_trn,x_tst,y_tst,N,q,T,start_time):
    np.random.seed(t)
    num_active_users = np.random.choice(np.arange(1, N), 1, False, q)
    active_users_ind = np.random.choice(np.arange(N), num_active_users, False)
    A_t = np.zeros(N)
    A_t[active_users_ind] = 1
    x_trn_active = x_trn[active_users_ind, :]
    y_trn_active = y_trn[active_users_ind]
    B_tst_t, w_tst_t = estimate_testing_loss_weight(x_trn_active, y_trn_active, x_tst, y_tst, regularizer, max_loss)
    if t% 1000 == 0:
        elapsed_time = time.time()-start_time
        print('%s out of %s' % (t, T))
        print('elapsed time is %s ' % elapsed_time)
    return A_t,B_tst_t,w_tst_t

num_cores = multiprocessing.cpu_count()
print('number of cores is %s'%num_cores)
start_time = time.time()
results = Parallel(n_jobs=num_cores)(delayed(paralell_sample_group_testing)(t,x_trn,y_trn,x_tst,y_tst,N,q,T,start_time) for t in range(T_new))
elapsed_time = time.time()-start_time
print('total elapsed time is %s ' % elapsed_time)
np.save('./iris/iris_result_balanced_reg1e-2.npy', results)
np.save('./iris/iris_time_balanced_reg1e-2.npy', elapsed_time)


## solve the feasibility problem to get approximate shapley values
results = np.load('iris/iris_result_unbalanced_reg1e-2_gt.npy')


A = np.vstack(results[:,0]).reshape((results.shape[0],results[0,0].shape[0]))
B_tst = results[:,1]
w = np.vstack(results[:,2])
results = None
C = {}
for i in range(N):
    for j in range(i+1,N):
        C[(i,j)] = Z/T_new*(B_tst.dot(A[:,i]-A[:,j]))
u_tot = estimate_testing_loss(x_trn, y_trn, x_tst, y_tst, regularizer, max_loss)

s = Variable(N)
constraints = [sum_entries(s)==u_tot]
for i in range(N):
    for j in range(i+1,N):
        constraints.append(s[i]-s[j]<= epsilon / (2 * np.math.sqrt(N)) + C[(i,j)])
        constraints.append(s[i] - s[j] >= -epsilon / (2 * np.math.sqrt(N)) + C[(i, j)])

prob = Problem(Minimize(0),constraints)
result = prob.solve(solver=SCS)
s_opt = s.value
np.save('./iris/sp_grouptest_unbalanced_reg1e-2_gt.npy',s_opt)