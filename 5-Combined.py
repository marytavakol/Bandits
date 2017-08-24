# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 14:45:39 2016

@author: maryam
"""

import Optimization
import numpy as np
import json
import math
import sys



class user:
    key = -1
    beta = []
    covMatrix = []
    cov_inv = []
    I = []
    
    def __init__(self, givenId):
        self.key = givenId
        self.beta = np.zeros((dim-2, 1))
        self.covMatrix = np.eye(dim-2)
        self.cov_inv = np.eye(dim-2)
        self.I = [0]*t
        

class arm:
    key = -1
    features = []
    covMatrix = []
    cov_inv = []
    theta = []
    I = []
    
    def __init__(self, givenId, fv):
        self.key = givenId
        self.features = np.reshape(fv, (dim-2, 1))
        self.covMatrix = np.eye(dim)
        self.cov_inv = np.eye(dim)
        self.theta = np.zeros((dim, 1))
        self.I = [0]*t
    
    def payoff(self, alpha, xt, beta, Au):
        zt = self.features
        variance = np.dot(np.dot(xt.T, self.cov_inv), xt)
        variance = variance + np.dot(np.dot(zt.T, Au), zt)
        cb = alpha * math.sqrt(variance)
        mean = np.dot(self.theta.T, xt) + np.dot(beta.T, zt)
        return mean + cb

def updateAll(at, ut, xt):
    global X
    global Xmult
    global Z
    global Zmult
    global r
    global Ia
    global Iu
    
    at.covMatrix = at.covMatrix + np.dot(xt, xt.T)
    at.cov_inv = np.linalg.inv(at.covMatrix)
    tempVec = np.dot(X, xt)
    Xmult = np.hstack((Xmult, tempVec))
    Xmult = np.vstack((Xmult, (np.append(tempVec, np.dot(xt.T, xt), 0)).T))
    X = np.vstack((X, xt.T))
    r = np.append(r, -reward)
    
    zt = at.features
    tempVec = np.dot(Z, zt)
    Zmult = np.hstack((Zmult, tempVec))
    Zmult = np.vstack((Zmult, (np.append(tempVec, np.dot(zt.T, zt), 0)).T))
    Z = np.vstack((Z, zt.T))
    ut.covMatrix = ut.covMatrix + np.dot(zt, zt.T)
    ut.cov_inv = np.linalg.inv(ut.covMatrix)
    
    for a in arms:
        a.I.append(0)
        if a == at:
            (at.I)[t-1] = 1
            temp = np.reshape(at.I, (t ,1))
            Ia = np.hstack((Ia, temp[:t-1]))
            Ia = np.vstack((Ia, temp.T))
        
    for u in users:
        u.I.append(0)
        if u == ut:
            (ut.I)[t-1] = 1
            temp = np.reshape(ut.I, (t ,1))
            Iu = np.hstack((Iu, temp[:t-1]))
            Iu = np.vstack((Iu, temp.T))
            
    #if method == 'cvxopt':
    P = np.eye(t) + np.multiply(Ia, Xmult) + np.multiply(Iu, Zmult)
    A = Optimization.CombinedSquareCvx(P, r, np.ones((1, t)), 0.0)
#    elif method == 'cplex':
#        P = np.multiply(Ia[t-1], Xmult[:, t-1]) + np.multiply(Iu[t-1], Zmult[:, t-1])
#        P[t - 1] += 1
#        A = Optimization.qpex(P, r, t)
#    elif method == 'GD':
#        A = Optimization.GD(P, np.reshape(r, (t, 1)), pre_a, t)
#    else:
#        P = np.eye(t) + np.multiply(Ia, Xmult) + np.multiply(Iu, Zmult)
#        A = Optimization.CombinedSquareSci(P, r, pre_a, method)
    
    return A



if __name__ == '__main__':
    if len(sys.argv) < 4:
        print ('enter the train(input) and model(item & user) file names')# + Optimization_method')
        #print('\n cvxopt - GD - BFGS - CG - Newton-CG - cplex\n')
        sys.exit(0)
    alpha = 2.36
    #method = str(sys.argv[4])
    arms = []
    aids = []
    users = []
    uids = []
    session = -1
    context = []
    print ('Personalized LinUCB in dual mode...')
    sys.stdout.flush()
    data = []
    with open(str(sys.argv[1]), "r") as fd:
        for line in fd:
            data.append(json.loads(line))
    
    dim = len(data[0]['features'])# + 1
    
    X = np.empty((0, dim))
    Xmult = np.empty((0, 0))
    Z = np.empty((0, dim - 2))
    Zmult = np.empty((0, 0))
    r = []
    Ia = np.empty((0, 0))
    Iu = np.empty((0, 0))
    #pre_a = np.empty((0, 1))
    
    #np.random.seed(379)
    t = 0
    for line in data:
        if line['item'] not in aids:
            aids.append(line['item'])
            arms.append(arm(line['item'], (line['features'])[:(dim-2)]))
        if line['user'] not in uids:
            uids.append(line['user'])
            users.append(user(line['user']))
                        
        ut = uids.index(line['user'])
        if line['session'] == session: #ongoing session
            t += 1
            if t % 400 == 0:
                print(t)
                sys.stdout.flush()
                
            xt = np.reshape(context, (dim, 1))
            ucb = [None] * len(arms)
            t_c = 0
            for a in arms:
                ucb[t_c] = a.payoff(alpha, xt, users[ut].beta, users[ut].cov_inv)
                t_c += 1
            
            maxucb = max(ucb)
            idx = [i for i, j in enumerate(ucb) if (maxucb-j) <= 0.000001] 
            rnd = np.random.randint(0, len(idx))
            at = idx[rnd]
            reward = -1
            if aids[at] == line['item']:
                reward = 1

            A = updateAll(arms[at], users[ut], xt)
            #pre_a = A
            for a in arms:
                a.theta = np.dot(np.multiply(X, np.reshape(a.I, (t, 1))).T, A)
            for u in users:
                u.beta = np.dot(np.multiply(Z, np.reshape(u.I, (t, 1))).T, A)
    
        context = line['features']#np.append([1], [line['features']])
        session = line['session']

    del data
    print ("Total number of arms: ", len(arms), " == ", len(aids))    
    sys.stdout.flush()
    with open(str(sys.argv[2]), "w") as fd:
        for arm in arms:
            line = {}
            line['key'] = arm.key
            line['features'] = arm.features.ravel().tolist()
            line['theta'] = arm.theta.ravel().tolist()
            json.dump(line, fd)
            fd.write('\n')
    with open(str(sys.argv[3]), "w") as fd:
        for u in users:
            line = {}
            line['key'] = u.key
            line['beta'] = u.beta.ravel().tolist()
            json.dump(line, fd)
            fd.write('\n')

    print ("Done!")