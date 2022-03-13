from utils.utils import *
import pandas as pd 
import numpy as np 

my_kfolds = import_kfolds("/Users/ngayulo/Documents/image-features/codes/utils/stratifiedkfolds.json")

my_data = pd.read_csv("/Users/ngayulo/Documents/image-features/archive/output_3/original/alexnet/alexnet_feature_activation.csv", index_col=0)
new_data = pd.read_csv("/Users/ngayulo/Documents/image-features/archive/output_3/cue-conflict/alexnet/alexnet_feature_activation.csv", index_col = 0)

new_indexes = new_data.index.tolist()
#print(len(new_indexes))
#print(new_indexes[0:15])

folds = {}

ref = {}
for key, val in my_kfolds.items():
    print(key)
    test_index = val['test']
    new_test = []
    new_train = []
    new_test_indexes = []
    new_train_indexes = []
    for i in my_data.index[test_index]:
        image = i[0:-4]
        for j in new_indexes:
            image_match = j.split("-")[0]
            if image_match == image:
                new_test.append(j)
    
    new_train = list(set(new_indexes)-set(new_test))

    new_test_indexes = [new_indexes.index(test) for test in new_test]
    new_train_indexes = [new_indexes.index(train) for train in new_train]

    print(len(new_test),len(new_train))
    #print(len(new_test_indexes),len(new_train_indexes))

    folds[key] = {}
    folds[key]['train'] = new_train_indexes
    folds[key]['test'] = new_test_indexes

    ref[int(val['test'][0])] = new_test_indexes
    
print(ref)
#dump folds to json
with open("/Users/ngayulo/Documents/image-features/codes/utils/reference.json", 'w') as fp:
    json.dump(ref, fp)
    


