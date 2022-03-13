# get folds and save as json file

from numpy.lib.shape_base import split
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from utils.utils import *

df = pd.read_csv("/Users/ngayulo/Documents/image-features/output/original/alexnet/alexnet_feature_activation.csv", index_col=0)
print(df.head())
X = df.iloc[:,0:-1].to_numpy()
Y = df.iloc[:,-1]

kf = StratifiedKFold(n_splits=5,shuffle=True, random_state=66)

for train,test in kf.split(X,Y):
    print(len(train),len(test))
    #print(Y[test])

save_kfolds_r(kf,X,Y,"style-transfer-20per_kfolds.json")
