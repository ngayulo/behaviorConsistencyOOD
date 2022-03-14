# gehiros's experiments

import torch 
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from features import *
#import probabilities_to_decision
import pandas as pd 
import numpy as np
from sklearn.preprocessing import normalize

device = "cuda" if torch.cuda.is_available() else "cpu"

image_paths = {
    'original': "input1/original/",
    'edges': "input1/edges/",
    'silhouette': "input1/silhouettes/",
    'cue-conflict': "input1/style-transfer/",
#    'grayscale': "input1/grayscale/"
}
#'vonenet_resnet50','vonenet_cornets
#'alexnet','resnet18', 'resnet34', 'resnet50', 'resnet101','vgg11','vgg13','vgg16','vgg19','densenet121','densenet169'
model_names = ['alexnet','resnet18', 'resnet34', 'resnet50', 'resnet101','vgg11','vgg13','vgg16','vgg19','densenet121','densenet169','vonenet_resnet50','vonenet_cornets']

exp_code = 'geirhos_3'

def get_response(path,model_name):
    print(model_name)
    fnc_name = f'get_{model_name}(inter=False)'
    model = eval(fnc_name)    
    filenames, input = image_process(path,model_name)
    model.to(device)
    input = input.to(device)
    print(input.shape)
    dataloader = DataLoader(input)
    response = []
    mapping = probabilities_to_decision.ImageNetProbabilitiesTo16ClassesMapping()
    first = True
    for batch in dataloader:
        batch_input = batch
        with torch.no_grad():
            batch_output = F.softmax(model(batch_input),dim=1)

        output = batch_output.cpu().numpy().reshape(-1,1)
        #print(output.shape)
        decision_from_16_classes, labels, prob = mapping.probabilities_to_decision(output)
        response.append(decision_from_16_classes)

        list = zip(labels,prob)
        sorted_list = sorted(list, key = lambda t: t[0])

        col, prob = zip(*sorted_list)
        prob = np.array(prob).reshape(1,-1)
        norm = normalize(prob,norm='l1',axis=1)
        #print(norm)
        if first==True:
            df_prob = norm
            first=False
        else:
            df_prob = np.concatenate((df_prob,norm),axis=0)
        
    df_prob = pd.DataFrame(df_prob,columns=col)
    df_prob['unique_img'] = filenames
    df_prob = df_prob.set_index('unique_img').sort_index()

    labels = get_labels(filenames,response)
    df = pd.DataFrame({'unique_img':filenames, 'object_response':response, 'category':labels})
    return df, df_prob

def response_extraction(exp_code,model,img_type):
    path = os.path.join(os.path.abspath(os.getcwd()),image_paths[img_type])
    outfile = os.path.join(os.path.abspath(os.getcwd()),f"{exp_code}/{img_type}/{model}/arg_max_responses/responses.csv")
    if not os.path.exists(outfile):
        os.makedirs(os.path.join(os.path.abspath(os.getcwd()),f"{exp_code}/{img_type}/{model}/arg_max_responses/"))
    out = open(outfile,'w')
    out.close()
    
    outfile_prob = os.path.join(os.path.abspath(os.getcwd()),f"{exp_code}/{img_type}/{model}/prob_responses/prob.csv")
    if not os.path.exists(outfile_prob):
        os.makedirs(os.path.join(os.path.abspath(os.getcwd()),f"{exp_code}/{img_type}/{model}/prob_responses/"))
    out = open(outfile_prob,'w')
    out.close()

    df, df_prob = get_response(path,model)
    print(df.head())
    print(df_prob.head())
    df.to_csv(outfile)
    df_prob.to_csv(outfile_prob)
    print("Done")
    return

def main():
    for model_name in model_names:
        for img_type in image_paths:
            response_extraction(exp_code,model_name,img_type)
    return

if __name__ == '__main__':
    main()
