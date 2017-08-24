# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 15:27:11 2016

@author: maryam
"""



import numpy as np
import json
import math
import sys



class user:
    key = -1
    beta = []
    A = []
    Ainv = []
    b = []
    B = []
    
    def __init__(self, givenId):
        self.key = givenId
        self.beta = np.zeros((dim, 1))
        self.A = np.eye(dim)
        self.Ainv = np.eye(dim)
        self.b = np.zeros((dim, 1))
        self.B = np.zeros((dim, dim))
    

    def update(self, z, reward):
        self.A = self.A + np.dot(z, z.T)
        self.Ainv = np.linalg.inv(self.A)
        self.B = self.B + np.dot(z, z.T)
        self.b = self.b + reward * z


class arm:
    key = -1
    features = []
    
    def __init__(self, givenId, fv):
        self.key = givenId
        self.features =  np.reshape(fv, (dim, 1))
        
    def payoff(self, theta, beta, Au, zCov_inv):
        variance = np.dot(np.dot(self.features.T, zCov_inv+Au), self.features)
        cb = alpha * math.sqrt(variance)
        mean = np.dot(self.features.T, beta+theta)
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
            
    dim = len(data[0]['features']) - 2 #dim of z
    A = np.eye(dim)
    Ai = np.eye(dim)
    zCov = np.eye(dim)
    zCov_inv = np.eye(dim)
    b = np.zeros((dim, 1))
    theta = np.zeros((dim, 1))
    arms = []
    ids = []
    users = []
    uids = []
    #session = -1
    print ('3- User Only + average method...')
    itr = 0
    #np.random.seed(379)
    for line in data:
        itr += 1
        if itr % 400 == 0:
            print (itr)
            sys.stdout.flush()
        
        if line['item'] not in ids:
            ids.append(line['item'])
            arms.append(arm(line['item'], (line['features'])[:(dim)]))
        if line['user'] not in uids:
            uids.append(line['user'])
            users.append(user(line['user']))
                      
        ut = uids.index(line['user'])
        #if line['session'] == session: #ongoing session
        ucb = [None] * len(arms)
        t_c = 0
        for a in arms:
            ucb[t_c] = a.payoff(theta, users[ut].beta, users[ut].Ainv, zCov_inv)
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
        A = A + np.dot(users[ut].B.T, np.dot(users[ut].Ainv, users[ut].B))
        b = b + np.dot(users[ut].B.T, np.dot(users[ut].Ainv, users[ut].b))
        users[ut].update(zt, reward)
        A = A + np.dot(zt, zt.T) - np.dot(users[ut].B.T, np.dot(users[ut].Ainv, users[ut].B))
        b = b + reward * zt - np.dot(users[ut].B.T, np.dot(users[ut].Ainv, users[ut].b))
        Ai = np.linalg.inv(A)
        theta = np.dot(Ai, b)
        for a in users:
            tmp = a.b - np.dot(a.B, theta)
            a.beta = np.dot(a.Ainv, tmp)
    
        
    
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
        line = {}
        line['theta'] = theta.ravel().tolist()
        json.dump(line, fd)
        fd.write('\n')
        for u in users:
            line = {}
            line['key'] = u.key
            line['beta'] = u.beta.ravel().tolist()
            json.dump(line, fd)
            fd.write('\n')

    print ("Done!")
