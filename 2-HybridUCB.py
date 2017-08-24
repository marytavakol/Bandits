# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 14:27:02 2016

@author: maryam
"""

import numpy as np
import json
import math
import sys


class arm:
    key = -1
    covMatrix = []
    cov_inv = []
    response = []
    theta = []
    B = []
    features = []
    
    def __init__(self, givenId, fv):
        self.key = givenId
        self.features =  np.reshape(fv, (fdim, 1))
        self.covMatrix = np.eye(dim)
        self.covMatrix[0,0] = 0
        self.cov_inv = np.eye(dim)
        self.response = np.zeros((dim, 1))
        self.B = np.zeros((dim, fdim))
        self.theta = np.zeros((dim, 1))
        
    def payoff(self, xt, Ai, beta):
        variance = np.dot(np.dot(xt.T, self.cov_inv), xt)
        variance = variance + np.dot(np.dot(self.features.T, Ai), self.features)
        cb = alpha * math.sqrt(variance)
        mean = np.dot(xt.T, self.theta) + np.dot(beta.T, self.features)
        return mean + cb

        
    def update(self, x, reward, beta):
        self.covMatrix = self.covMatrix + np.dot(x, x.T)
        self.cov_inv = np.linalg.inv(self.covMatrix)
        self.B = self.B + np.dot(x, (self.features).T)
        self.response = self.response + reward * x
        

        
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print ('enter the train(input) and model(output) file names.')
        sys.exit(0)
    
    alpha = 2.36    
    data = []
    with open(str(sys.argv[1]), "r") as fd:
        for line in fd:
            data.append(json.loads(line))
            
    dim = len(data[0]['features']) + 1
    fdim = dim-3
    A = np.eye(fdim)
    Ai = np.eye(fdim)
    zCov = np.eye(fdim)
    zCov_inv = np.eye(fdim)
    b = np.zeros((fdim, 1))
    beta = np.zeros((fdim, 1))
    arms = []
    ids = []
    session = -1
    context = []
    print ('2- LinUCB + average method...')
    itr = 0
    #np.random.seed(379)
    for line in data:
        itr += 1
        if itr % 400 == 0:
            print (itr)
            sys.stdout.flush()
        
        if line['item'] not in ids:
            ids.append(line['item'])
            arms.append(arm(line['item'], (line['features'])[:(dim-3)]))
                        
        if line['session'] == session: #ongoing session
            xt = np.reshape(context, (dim, 1))
            ucb = [None] * len(arms)
            t_c = 0
            for a in arms:
                ucb[t_c] = a.payoff(xt, zCov_inv, beta)
                t_c += 1
                    
            maxucb = max(ucb)
            idx = [i for i, j in enumerate(ucb) if (maxucb-j) <= 0.000001]
            rnd = np.random.randint(0, len(idx))
            at = idx[rnd]
            reward = -1
            if ids[at] == line['item']:
                reward = 1
            
            zt = arms[at].features
            zCov = zCov + np.dot(zt, zt.T)
            zCov_inv = np.linalg.inv(zCov)
            A = A + np.dot(arms[at].B.T, np.dot(arms[at].cov_inv, arms[at].B))
            b = b + np.dot(arms[at].B.T, np.dot(arms[at].cov_inv, arms[at].response))
            arms[at].update(xt, reward, beta)
            A = A + np.dot(zt, zt.T) - np.dot(arms[at].B.T, np.dot(arms[at].cov_inv, arms[at].B))
            b = b + reward * zt - np.dot(arms[at].B.T, np.dot(arms[at].cov_inv, arms[at].response))
            Ai = np.linalg.inv(A)
            beta = np.dot(Ai, b)
            for a in arms:
                tmp = a.response - np.dot(a.B, beta)
                a.theta = np.dot(a.cov_inv, tmp)
        
        context = np.append([1], [line['features']])
        session = line['session']
    
    del data        
    print ("Total number of arms: ", len(arms), " == ", len(ids))    
    with open(str(sys.argv[2]), "w") as fd:
        line = {}
        line['beta'] = beta.ravel().tolist()
        json.dump(line, fd)
        fd.write('\n')
        for arm in arms:
            line = {}
            line['key'] = arm.key
            line['features'] = arm.features.ravel().tolist()
            line['theta'] = arm.theta.ravel().tolist()
            json.dump(line, fd)
            fd.write('\n')

    print ("Done!")
