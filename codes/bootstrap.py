# bootstraping and signficance testing 

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import scipy.stats as st
import numpy as np

def bootstrap_confidence_int(vec1,vec2):
    n = len(vec1)
    arr = []
    for i in range(5000):
        samp_index = np.random.choice(np.arange(n),n,replace=True)
        r = pearsonr(vec1[samp_index],vec2[samp_index])[0]
        arr.append(r)
    conf = np.percentile(arr,[2.5,97.5])
    #plt.hist(arr,bins=np.arange(0,1.1,0.01))
    #plt.show()
    return conf

def bootstrap_std(vec1,vec2):
    n = len(vec1)
    arr = []
    for i in range(5000):
        samp_index = np.random.choice(np.arange(n),n,replace=True)
        r = pearsonr(vec1[samp_index],vec2[samp_index])[0]
        arr.append(r)
    std = np.nanstd(arr)
    return std

def significance(pair1,pair2,ceiling=1):
    assert len(pair1[0])==len(pair1[1])==len(pair2[0])==len(pair2[1])
    n = len(pair1[0])
    #print(n)
    r1_obs = pearsonr(pair1[0],pair1[1])[0]/np.sqrt(ceiling)
    r2_obs = pearsonr(pair2[0],pair2[1])[0]/np.sqrt(ceiling)
    d_obs = r1_obs - r2_obs
    #print(r1_obs,r2_obs,d_obs)
    diff = []
    for i in range(5000):
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        for j in range(n):
            factor = random.randint(1,10)
            if factor>5:
                x1.append(pair1[0][j])
                y1.append(pair1[1][j])
                x2.append(pair2[0][j])
                y2.append(pair2[1][j])
            else:
                x1.append(pair2[0][j])
                y1.append(pair2[1][j])
                x2.append(pair1[0][j])
                y2.append(pair1[1][j])
        assert len(x1)==len(x2)==len(y1)==len(y2)
        
        r1 = pearsonr(x1,y1)[0]
        r2 = pearsonr(x2,y2)[0]
        d = r1-r2
        diff.append(d)

    sum = 0 
    for item in diff:
        if abs(item) >= abs(d_obs):
            sum = sum + 1
    p = sum/5000
    
    return p
