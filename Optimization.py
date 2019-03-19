# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 10:17:17 2016

@author: maryam
"""

import numpy as np
from cvxopt import matrix
from cvxopt import solvers
#from scipy.optimize import minimize
import cplex


def SimpleSquare(X, rew):
    T = np.shape(X)[0]
    C = 1
    P = matrix(X + np.eye(T))
    qq = matrix(np.multiply(-C, rew))
    
    sol = solvers.qp(P, qq)
    
    return sol['x']
    
    
def SimpleLogistic(lam, X, rew):
    T = np.shape(X)[0]
    C = lam/1
    y = np.reshape(rew, (T, 1))
    #A = train_dual_cg_2(X, y, C)
    Ahat = train_dual_cg((np.multiply(X, y)).T, C) # orig lambda*T
    A = np.multiply(Ahat, y) * C
    return A
        
def GD(P, q, x, t):
    lr = 0.001
    x = np.reshape(np.append(x, 0), (t, 1))
    pre = qp(x, P, q)
    while True:
        g = np.dot(P, x) + q
        x = x - lr * g
        f = qp(x, P, q)
        if f >= pre:
            break
        pre = f
    return x + lr * g
    
c = cplex.Cplex()
c.set_log_stream(None)
c.set_warning_stream(None)
c.set_results_stream(None)
c.objective.set_sense(c.objective.sense.minimize)
#@profile
def qpex(P, q, t):
    global c
    c.variables.add(obj=[q[t-1]], lb=[-cplex.infinity])
    #ind = list(range(t))
    #for i in range(t):
    #    qmat.append([ind, P[i, :].tolist()])
    for i in range(t):
        c.objective.set_quadratic_coefficients(t-1, i, P[i])
    for i in range(t):
        c.objective.set_quadratic_coefficients(i, t-1, P[i])
    c.solve()
    return c.solution.get_values()

def CombinedSquareCvx(P, Q, A, b):
    sol = solvers.qp(matrix(P), matrix(Q), None, None, matrix(A), matrix(b))
    return sol['x']


def qp(x, P, Q):
    return 0.5 * np.dot(x.T, np.dot(P, x)) + np.dot(Q.T, x)
    
def qp_der(x, P, q):
    return np.dot(P, x) + q
    
def qp_hes(x, P, q):
    return P

#def CombinedSquareSci(P, Q, x0, mtd):
#    x0 = np.append(x0, 0)
#    res = minimize(qp, x0, args=(P, Q), method=mtd, jac=qp_der, hess=qp_hes)#, options={'disp': True})
#    return res.x
    
    
    
def ClosedForm(lam, X, rew):
    T = np.shape(X)[0]
    Cinv = T/lam
    B = np.dot(X, X.T) + np.multiply(np.eye(T), Cinv)
    qq = matrix(np.reshape(rew, (T, 1)))
    return np.dot(np.linalg.inv(B), qq)
    
    
def cg_dir(old_dir, grad, old_grad):
    delta = grad - old_grad
    beta = np.divide(np.dot(-grad.T, delta), np.dot(old_grad.T, old_grad))
    #beta = np.dot(grad.T, delta) / np.dot(old_dir.T, delta)
    
    direction = grad - beta * old_dir
    return direction
    
    
def train_dual_cg(x, v):
    c = np.dot(x.T, x)    # n x 1

    eps = np.finfo(float).eps    
    d, n = np.shape(x)
    alpha = np.tile(1.0/(n+1), (n, 1))
    old_g = np.zeros((d, 1))
    u = np.zeros((d, 1))    
    
    for itr in range(2000):
        old_alpha = alpha
        g = v*(np.dot(c, alpha)) + np.log(np.divide(alpha, (1-alpha)))
        #cons = np.where((alpha <= eps) & (g > 0)) | ((alpha >= (1-eps)) & (g < 0))
        cons = np.where((alpha <= eps) | (alpha >= (1-eps)))
        g[cons] = 0
        if itr == 0:
            u = g
        else:
            u = cg_dir(u, g, old_g)
        
        u[cons] = 0 # n x 1
        ug = np.dot(u.T, g) # 1 x 1 
        uhu = v*(np.dot(np.dot(u.T, c),u)) + np.sum(np.divide(np.divide(np.square(u), alpha), (1-alpha))) # 1 x 1
        ipos = np.where(u > 0)
        ineg = np.where(u < 0)
        step_max = min(np.append(np.divide(alpha[ipos], u[ipos]), np.divide(alpha[ineg]-1, u[ineg])))
        step_min = max(np.append(np.divide(alpha[ineg], u[ineg]), np.divide(alpha[ipos]-1, u[ipos])))
        step = ug/uhu
        if step > step_max:
            step = step_max
        elif step < step_min:
            step = step_min
        
        alpha = alpha - step*u
        old_g = g
        
        i = np.where(alpha < eps)
        alpha[i] = eps
        i = np.where(alpha > 1-eps)
        alpha[i] = 1 - eps
        
        if max(abs(alpha-old_alpha)) < 1e-8:
            break
    if itr > 1999:
        print('here')
    return alpha
