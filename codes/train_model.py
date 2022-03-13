# regular training decoder

from utils.utils import *
#from saved_files import *
from regression import *

img_types = ['original','edges','silhouette','cue-conflict']

models = {
    "vonenet_resnet50":{"epoch":200,"lr":0.1},
    "vonenet_cornets":{"epoch":200,"lr":0.1}
}


kf_file = input("Enter kfold file: ")
exp_code = input("Enter experiment code: ")

import os 
for model,val in models.items():
    i=0
    for i in range(10):
        for img in img_types:
            saving_path_responses = os.path.join(os.path.abspath(os.getcwd()),f"{exp_code}/{img}/{model}/responses/")
            activations = os.path.join(os.path.abspath(os.getcwd()),f"{exp_code}/{img}/{model}/{model}_feature_activation.csv")
            saving_path_decoder = os.path.join(os.path.abspath(os.getcwd()),f"{exp_code}/{img}/{model}/models/decoder_{i}/")

            if not os.path.exists(saving_path_responses):
                os.makedirs(saving_path_responses)
            if not os.path.exists(saving_path_decoder):
                os.makedirs(saving_path_decoder)

            print(model, img)
            print(activations)

            if img == 'cue-conflict':
                kfold_file = os.path.join(os.path.abspath(os.getcwd()),"utils/style-transfer-"+kf_file)
                kf = import_kfolds(kfold_file)
            else:
                kfold_file = os.path.join(os.path.abspath(os.getcwd()),"utils/"+kf_file)
                kf = import_kfolds(kfold_file)
            
            df = cross_output(activations,kf,saving_path_decoder,val['epoch'],val['lr'])
            df.to_csv(saving_path_responses+f"{model}_{i}.csv")
