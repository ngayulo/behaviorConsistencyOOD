import json

def import_kfolds(filename):
    #load to dict to be used
    with open(filename) as f:
        kfolds = json.load(f)
    return kfolds

def import_models_parameters(filename):
    with open(filename) as f:
        models = json.load(f)
    return models

def save_kfolds(kf,X,Y,outfile):
    folds = {}
    count = 1
    for train_index, test_index in kf.split(X,Y):
        folds['fold_{}'.format(count)] = {}
        folds['fold_{}'.format(count)]['train'] = train_index.tolist()
        folds['fold_{}'.format(count)]['test'] = test_index.tolist()
        count += 1
    #dump folds to json
    with open(outfile, 'w') as fp:
        json.dump(folds, fp)
    return
    
def save_kfolds_r(kf,X,Y,outfile):
    folds = {}
    count = 1
    for test_index, train_index in kf.split(X,Y):
        folds['fold_{}'.format(count)] = {}
        folds['fold_{}'.format(count)]['train'] = train_index.tolist()
        folds['fold_{}'.format(count)]['test'] = test_index.tolist()
        count += 1
    #dump folds to json
    with open(outfile, 'w') as fp:
        json.dump(folds, fp)
    return 

