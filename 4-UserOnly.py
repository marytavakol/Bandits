# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 13:39:19 2016

@author: maryam
"""



import numpy as np
import json
import sys
import math


class user:
    key = -1
    beta = []
    A = []
    Ainv = []
    b = []
    
    def __init__(self, givenId):
        self.key = givenId
        self.beta = np.zeros((dim, 1))
        self.A = np.eye(dim)
        self.A[0,0] = 0
        self.Ainv = np.eye(dim)
        self.b = np.zeros((dim, 1))
    

    def update(self, z, reward):
        self.A = self.A + np.dot(z, z.T)
        self.Ainv = np.linalg.inv(self.A)
        self.b = self.b + reward * z
        self.beta = np.dot(self.Ainv, self.b)
    

class arm:
    key = -1
    features = []
    
    def __init__(self, givenId, fv):
        self.key = givenId
        self.features =  np.reshape(np.append([1], [fv]), (dim, 1))        
        
    def payoff(self, beta, Au):
        variance = np.dot(np.dot(self.features.T, Au), self.features)
        cb = alpha * math.sqrt(variance)
        mean = np.dot(beta.T, self.features)
        return mean + cb
        
     
        
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print ('enter the train(input) and model(item & user) file names.')
        sys.exit(0)
      
    alpha = 2.36
    data = []
    with open(str(sys.argv[1]), "r") as fd:
        for line in fd:
            data.append(json.loads(line))
    
    dim = len(data[0]['features']) - 1
    
    arms = []
    ids = []
    users = []
    uids = []
    session = -1
    t = 0
    print ('4- User only method...')
    #np.random.seed(379)
    for line in data:
        if line['item'] not in ids:
            ids.append(line['item'])
            arms.append(arm(line['item'], (line['features'])[:(dim-1)]))
        if line['user'] not in uids:
            uids.append(line['user'])
            users.append(user(line['user']))
              
        ut = uids.index(line['user'])
        #if line['session'] == session: #ongoing session
        t += 1
        if t % 400 == 0:
            print (t)
            sys.stdout.flush()
        
        ucb = [None] * len(arms)
        t_c = 0
        for a in arms:
            ucb[t_c] = a.payoff(users[ut].beta, users[ut].Ainv)
            t_c += 1

        maxucb = max(ucb)
        idx = [i for i, j in enumerate(ucb) if (maxucb-j) <= 0.000001]
        rnd = np.random.randint(0, len(idx))
        at = idx[rnd]
        reward = -1
        if ids[at] == line['item']:
            reward = 1
        
        users[ut].update(arms[at].features, reward)
        
        #session = line['session']
    
    del data        
    print ("Total number of arms: ", len(arms), " == ", len(ids))    
    with open(str(sys.argv[2]), "w") as fd:
        for arm in arms:
            line = {}
            line['key'] = arm.key
            line['features'] = arm.features.ravel().tolist()
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
