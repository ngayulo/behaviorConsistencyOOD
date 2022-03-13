# behavior-metrics

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import pearsonr 
import random

# get probability matrix from dataframe of responses (for human data)
def probability_matrix(df):
    images = sorted(pd.unique(df['unique_img']))
    classes = sorted(pd.unique(df['category']))
    prob_mat = np.zeros((len(images), len(classes)))

    # for each unique image, count the number of responses for each class
    for r in range(len(images)):
        ind = df [ (df['unique_img']== images[r]) ].index
        for c in range(len(classes)):
            c_count = 0
            for i in ind:
                #print(df.loc[i,'object_response'])
                if df.loc[i,'object_response'] == classes[c]:
                    c_count += 1
            prob_mat[r,c] = c_count/len(ind)
    df = pd.DataFrame(prob_mat, index = images, columns = classes)
    return df

# metric object, initiated by the name of the metric you use, returns the behavior signature 
class metric():
    def __init__(self,name):
        self.metric_name = name
    
    def get_metric_performance(self,prob_mat,call_metric=None):
        if call_metric==None:
            if self.metric_name == 'O1':
                return self.object_1(prob_mat)
            elif self.metric_name == 'I1':
                return self.image_1(prob_mat,normalize=False)
            elif self.metric_name == 'I1N':
                return self.image_1(prob_mat)
        else:
            if call_metric == 'O1':
                return self.object_1(prob_mat)
            elif call_metric == 'I1':
                return self.image_1(prob_mat,normalize=False)
            elif call_metric == 'I1N':
                return self.image_1(prob_mat)

    # input: probability matrix
    # output: labels, accuracy vector (auc_vec), dprime vector (dp_vec)
    def object_1(self,prob_mat):
        cat = prob_mat.columns.tolist()
        auc_vec = np.zeros(len(cat))
        dp_vec = np.zeros(len(cat))

        for c in range(len(cat)):
            ind = get_belonging_images(prob_mat.index,cat[c])
            t_values = prob_mat.loc[ind,cat[c]].astype(np.float) # hit rate
            auc_vec[c] = np.nanmean(t_values) # average hit rate for accuracy
            # drop images that belong to the category
            f_values = prob_mat.drop(index=ind).loc[:,cat[c]].astype(np.float) # false alarm rate
            dp_vec[c] = zscore(np.nanmean(t_values)) - zscore(np.nanmean(f_values))  # dprime

        dp_vec = np.clip(dp_vec,-5,5)
        return cat, auc_vec, dp_vec

    # input: probability matrix, normalize default to True
    # output: labels, accuracy vector (auc_vec), dprime vector (dp_vec)
    def image_1(self,prob_mat,normalize=True):
        images = prob_mat.index.tolist()
        auc_vec = np.zeros(len(images))
        dp_vec = np.zeros(len(images))
        labels = []

        for i in range(len(images)):
            c = get_belonging_category(prob_mat.columns,images[i])
            t_values = prob_mat.loc[images[i],c] # hit rate
            auc_vec[i] = t_values
            labels.append(c)
            ind = get_belonging_images(prob_mat.index,c)
            # drop images that belong to the category
            f_values = prob_mat.drop(index=ind).loc[:,c].astype(np.float) # false alarm rate 
            dp_vec[i] = zscore(t_values) - zscore(np.nanmean(f_values))
        #print(vec)

        # if normalize, get object level accuracy and subtract from image level vector 
        def auc_subtract_mean():
            normauc_vec = np.zeros(len(auc_vec))
            cat, sub = self.object_1(prob_mat)[:-1]
            for i in range(len(auc_vec)):
                c = get_belonging_category(cat,images[i])
                normauc_vec[i] = auc_vec[i] - sub[cat.index(c)]
            return normauc_vec

        dp_vec = np.clip(dp_vec,-5,5) # clip dprimes at -5 and 5
        # normalize dprime 
        def dp_subtract_mean():
            tmp = pd.DataFrame({'Images':images,'Labels':labels,'dprimes':dp_vec})
            normdp_vec = np.zeros(len(dp_vec))
            for i in range(len(dp_vec)):
                cat = tmp['Labels'][i]
                mu = tmp[ tmp['Labels']==cat ]['dprimes'].mean()
                normdp_vec[i] = dp_vec[i] - mu
            return normdp_vec
        
        if normalize == True:
            normauc_vec = auc_subtract_mean()
            normdp_vec = dp_subtract_mean()
            return images,normauc_vec, normdp_vec
        else:
            return images,auc_vec, dp_vec

# input: names of images and a category
# output: list of images that belongs to the category
# this function works only if the image's name contains the category label
def get_belonging_images(indexes,c):
    ind = []
    for i in indexes:
        category = i.split('-')[0]
        if category.find(c)!=-1:
            ind.append(i)
    return ind

# input: list of categories and image name
# output: category that the image belongs to 
# this function works only if the image's name contains the category label
def get_belonging_category(category_list,i):
    for c in category_list:
        name = i.split('-')[0]
        if name.find(c)!=-1:
            return c

from scipy.stats import norm
def zscore(x):
    return np.clip(norm.ppf(x),-100,100)


class error_consistency(metric):
    def __init__(self):
        self.obs_consistency = 0
        self.exp_consistency = 0  
        self.kappa = 0  

    # calling function for c_obs
    def get_obs_consistency(self):
        return self.obs_consistency

    # calling function for c_exp
    def get_exp_consistency(self):
        return self.exp_consistency

    # calling function for kappa
    def get_kappa(self,subj1,subj2):
        self.calculate_exp_consistency(probability_matrix(subj1),probability_matrix(subj2))
        self.calculate_obj_consistency(subj1,subj2)
        #print('expected:',self.exp_consistency)
        #print('observed:',self.obs_consistency)
        if float(self.obs_consistency)==1.0:
            self.kappa = 1.0
        else:
            self.kappa = (self.obs_consistency - self.exp_consistency) / (1-self.exp_consistency)
        #print('kappa:',self.kappa)
        return self.kappa

    # Take subject's responses for each trial and determine how many times they had the same response
    def calculate_obj_consistency(self,subj1,subj2):
        subj1 = subj1.sort_values(by='unique_img')
        subj2 = subj2.sort_values(by='unique_img')
        # get the number of trials
        if subj1.shape[0] == subj2.shape[0]:
            n =  subj1.shape[0]
        else:
            return 
    
        equal = 0
        sub1_cat = subj1['category'].tolist()
        sub2_cat = subj2['category'].tolist()
        sub1_resp = subj1['object_response'].tolist()
        sub2_resp = subj2['object_response'].tolist()
        for i in range(n):
            if sub1_cat[i] == sub2_cat[i]:
                category = sub1_cat[i]
                if sub1_resp[i] == category and sub2_resp[i] == category:
                    equal += 1
                elif sub1_resp[i] != category and sub2_resp[i] != category:
                    equal += 1
        
        self.obs_consistency = equal/n
        return 0

    # get the accuracy of the subject using 'O1' metric, which averages the accuracies across object category 
    def calculate_exp_consistency(self,subj1_prob,subj2_prob):
        #print(subj1_prob)
        #print(subj2_prob)
        vec1 = self.get_metric_performance(subj1_prob,'O1')[1]
        vec2 = self.get_metric_performance(subj2_prob,'O1')[1]
        #print(vec1)
        #print(vec2)
        p_i = np.nanmean(vec1)
        p_j = np.nanmean(vec2)
        
        # expected_consistency formula
        self.exp_consistency = (p_i*p_j) + ((1-p_i) * (1-p_j))
        return 0
    
# for checking if the list contains all constant
def check_elements(l):
    return len(set(l)) == 1

# calculate correlation
def correlation(x,y):
    if check_elements(x)==True or check_elements(y)==True:
        return np.nan, np.nan
    else:
        return pearsonr(x,y)

##########################################################################################

def split_correlation(df,metric_name,my_split):
    half1, half2 = my_split.equal_halves(df)
    
    prob_m1 = probability_matrix(half1)
    prob_m2 = probability_matrix(half2)
    
    img1, auc1, dp1 = metric(metric_name).get_metric_performance(prob_m1)
    img2, auc2, dp2 = metric(metric_name).get_metric_performance(prob_m2)
    
    auc_r = correlation(auc1,auc2)[0]
    dp_r = correlation(dp1,dp2)[0]
    if np.isnan(auc_r) or np.isnan(dp_r):
        return split_correlation(df,metric_name,my_split)
    else:
        return auc_r, dp_r


# main ceiling calculation for human subject
# input: dataframe of all subject responses, metric name
# output: mean ceiling of accuracy, std ceiling of accuracy, mean ceiling of dprime, std ceiling of dprime 
def ceiling(df,metric_name):
    auc_c = []
    dp_c = []
    my_split = split()
    for i in range(10):
        auc, dp = split_correlation(df,metric_name,my_split)
        auc_c.append(auc)
        dp_c.append(dp)
    
    return np.nanmean(auc_c),np.nanstd(auc_c), np.nanmean(dp_c), np.nanstd(dp_c)

###########################################################################################
# Getting equal halves
def my_fun(x):
    return np.sin(x)


# class to get random splits, seeded
class split():
    def __init__(self):
        self.counter = 1

    def equal_halves(self,df):
        subj_list = pd.unique(df['subj'])
        random.seed(self.counter)
        random.shuffle(subj_list)
        self.counter += 1

        if len(subj_list)%2==0:
            ind1, ind2 = self.even_split(df,subj_list)
        else:
            ind1, ind2 = self.odd_split(df,subj_list)

        half1 = df.loc[ind1,:]
        half2 = df.loc[ind2,:]

        print("Half1:", pd.unique(half1['subj']))
        print("Half2:", pd.unique(half2['subj']))
        return half1, half2

    def even_split(self,df,subjects):
        ind1 = []
        ind2 = []
        random.seed(my_fun(self.counter))
        group1 = random.sample(set(subjects),len(subjects)//2)
        group2 = list(set(subjects)-set(group1))

        for s in group1:
            ind1.extend(df[ df['subj']==s ].index)
        for s in group2:
            ind2.extend(df[ df['subj']==s ].index)
        return ind1,ind2

    def odd_split(self,df,subjects):
        ind1 = []
        ind2 = []

        ind1, ind2 = self.even_split(df,subjects[:-1])
        ind1.extend(df[ df['subj']==subjects[-1] ].index)
        ind2.extend(df[ df['subj']==subjects[-1] ].index)

        return ind1,ind2
