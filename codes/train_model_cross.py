# train decoder on original, test on edges,silhouette, cue-conflict
from utils.utils import * 
from regression import *
from sklearn import preprocessing
import os 

train_code = 'same_domain'
exp_code = 'cross_domain'

# Get test sets
kfold_file = os.path.join(os.path.abspath(os.getcwd()),"utils/stratifiedkfolds.json")
kfolds = import_kfolds(kfold_file)
stnd_test_index = []

for key, val in kfolds.items():
    #print(key)
    stnd_test_index.append(val['test'])

# test sets for style-transfer
st_kfold_file = os.path.join(os.path.abspath(os.getcwd()),"utils/style-transfer-stratifiedkfolds.json")
st_kfolds = import_kfolds(st_kfold_file)
spec_test_index = []

for key, val in st_kfolds.items():
    #print(key)
    spec_test_index.append(val['test'])

import pandas as pd
import numpy as np 
def evaluate(zip_list,activation_file,first = True):
    print("Evaluating...")
    # read file of data
    df = pd.read_csv(activation_file,index_col=0)
    df_resp = []
    auc = []
    for pretrained_model, test_index in zip_list:
        #print(pretrained_model)

        # get test set
        data = df.iloc[:,0:-1].to_numpy()[test_index]
        labels = df.iloc[:,-1][test_index]
        le = preprocessing.LabelEncoder()
        labels = le.fit_transform(labels)

        model = Log_Model(data.shape[1],16)
        model.load_state_dict(torch.load(pretrained_model))
        model.eval()

        img_labels = df.index[test_index].values.reshape(-1,1)
        test_data = Dataset(data,labels)
        testloader = DataLoader(test_data, batch_size=len(test_data))
        
        for batch in testloader:
            data,labels = batch
            # Get predictions
            out = model(data.float()) 
        acc = accuracy(out,labels)
        auc.append(acc)
        # Apply softmax for each output_ff row to get probabilities
        probs = F.softmax(out, dim=1) 
        n = probs.detach().numpy()
        
        df_k = np.concatenate((img_labels, n),axis=1)
        if first == True:
            df_resp = df_k
            first = False
        else:
            df_resp = np.concatenate((df_resp, df_k))

    columns = ["unique_img", "airplane", "bear", "bicycle", "bird", "boat", "bottle", "car", "cat", "chair", "clock", "dog", "elephant", "keyboard", "knife", "oven", "truck"]
    df_resp = pd.DataFrame(df_resp, columns=columns).sort_values(by="unique_img").set_index("unique_img")
    
    #print(df_resp.head())  
    return df_resp, np.nanmean(auc)


def main():
    #m = np.zeros((4,4))
    ###########################################################################################
    # Get file names
    #model = input("Enter model name: ")
    model_list = ['vonenet_resnet50','vonenet_cornets']
    trained_sets = ['original'] #,'edges','silhouette','grayscale','style-transfer']
    test_sets = ['edges','silhouette','cue-conflict']
    ###########################################################################################

    for model in model_list:
        for key_train in trained_sets:
            path_train = os.path.join(os.path.abspath(os.getcwd()),f"{train_code}/{key_train}/{model}/models/")
            for root,subdir,files in os.walk(path_train):
                print(subdir)
                for sub in subdir:
                    decoder_path = os.path.join(path_train,sub)  
                    print(decoder_path)  
                    for root,subdir,files in os.walk(decoder_path):
                        model_list = sorted([os.path.join(root,file) for file in files if file.endswith('.pth') ])

                        for key_test in test_sets:

                            saving_dir = os.path.join(os.path.abspath(os.getcwd()),f"{exp_code}/{key_test}/{model}/prob_responses/")
                            if not os.path.exists(saving_dir):
                                os.makedirs(saving_dir)

                            activation = os.path.join(os.path.abspath(os.getcwd()),f"{train_code}/{key_test}/{model}/{model}_feature_activation.csv")
                            print("Trained on:", key_train,decoder_path)
                            print("Test on:", key_test, activation)

                            # get test set depending on test image type
                            if key_test == 'edges' or key_test == 'silhouette':
                                print("Using standard test set")
                                sets = zip(model_list,stnd_test_index)
                            elif key_test == 'cue-conflict':
                                print("Using special test set")
                                sets = zip(model_list,spec_test_index)

                            df_m, auc = evaluate(sets,activation)  
                            df_m.to_csv(os.path.join(saving_dir,f'{sub}_trained_{key_train}_tested_{key_test}.csv'))
                            print(auc)
    return

if __name__=="__main__":
    main()
