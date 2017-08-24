# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 11:48:28 2015

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
    
    def __init__(self, givenId):
        self.key = givenId
        self.covMatrix = np.eye(dim)
        self.covMatrix[0,0] = 0
        self.cov_inv = np.eye(dim)
        self.response = np.zeros((dim, 1))
        self.theta = np.zeros((dim, 1))
    
    def payoff(self, alpha, x):
        variance = np.dot(np.dot(x.T, self.cov_inv), x)
        CB = alpha * math.sqrt(variance)
        mean = np.dot(self.theta.T, x)
        return mean + CB
        
    def update(self, x, reward):
        self.covMatrix = self.covMatrix + np.dot(x, x.T)
        self.cov_inv = np.linalg.inv(self.covMatrix)
        self.response = self.response + reward * x
        self.theta = np.dot(self.cov_inv, self.response)
        
        

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print ('enter the train(input) and model(output) file names.')
        sys.exit(0)
    alpha = 2.36
    arms = []
    ids = []
    session = -1
    context = []
    print ('1- LinUCB method...')
    itr = 0
    data = []
    with open(str(sys.argv[1]), "r") as fd:
        for line in fd:
            data.append(json.loads(line))
     
    dim = len(data[0]['features']) + 1
    #np.random.seed(379)
    for line in data:
        itr += 1
        if itr % 400 == 0:
            print (itr)
            sys.stdout.flush()
            
        if line['item'] not in ids:
            ids.append(line['item'])
            arms.append(arm(line['item']))
                        
        if line['session'] == session: #ongoing session
            xt = np.reshape(context, (dim, 1))
            ucb = np.zeros((len(arms), 1))
            t_c = 0
            for a in arms:
                ucb[t_c] = a.payoff(alpha, xt)
                t_c += 1
                
            maxucb = np.max(ucb)
            idx = [i for i, j in enumerate(ucb) if (maxucb-j) <= 0.000001]
            rnd = np.random.randint(0, len(idx))
            at = idx[rnd]
            reward = -1
            if ids[at] == line['item']:
                reward = 1
            arms[at].update(xt, reward)
            
        context = np.append([1], [line['features']])
        session = line['session']
    
    del data
    print ("Total number of arms: ", len(arms), " == ", len(ids))    
    with open(str(sys.argv[2]), "w") as fd:
        for arm in arms:
            line = {}
            line['key'] = arm.key
            line['theta'] = arm.theta.ravel().tolist()
            json.dump(line, fd)
            fd.write('\n')

    print ("Done!")
