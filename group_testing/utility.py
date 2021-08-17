import numpy as np
from cvxpy import *

# logistic regression by cvx
def estimate_testing_loss_weight(x_trn, y_trn, x_tst, y_tst, regularizer, max_loss):
    w = Variable(x_trn.shape[1])
    n_trn = x_trn.shape[0]
    loss = 0
    for ni in range(n_trn):
        loss += logistic(-y_trn[ni]*x_trn[ni,:]@w)
    prob = Problem(Minimize(1 / n_trn * loss + regularizer / 2 * sum_squares(w)))
    result = prob.solve(solver=SCS)
    w_value = np.squeeze(np.array(w.value))
    n_tst = x_tst.shape[0]
    loss_test = 1 / n_tst * np.sum(np.log(1 + np.exp(-y_tst * x_tst.dot(w_value))))
    util = max_loss - loss_test
    if util < 0:
        return 0, w_value
    else:
        return util, w_value



