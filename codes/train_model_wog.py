# train decoder on original and some o.o.d. stimuli

from utils.utils import *
from regression import *

img_types = ['silhouette','cue-conflict']

models = {
    "vonenet_resnet50":{"epoch":200,"lr":0.1},
    "vonenet_cornets":{"epoch":200,"lr":0.1}
}

#param_file = input("Enter paramerter file: ")
kf_file = input("Enter kfold file: ")
exp_code = input("Enter experiment code: ")

#models = import_models_parameters(param_file)

for img in img_types:
    for model,val in models.items():
        activations_og = os.path.join(os.path.abspath(os.getcwd()),f"same_domain/original/{model}/{model}_feature_activation.csv")
        activations = os.path.join(os.path.abspath(os.getcwd()),f"same_domain/{img}/{model}/{model}_feature_activation.csv")
        for i in range(10):
            saving_path_responses = os.path.join(os.path.abspath(os.getcwd()),f"{exp_code}/{img}/{model}/prob_responses/")
            saving_path_decoder = os.path.join(os.path.abspath(os.getcwd()),f"{exp_code}/{img}/{model}/models/decoder_{i}/")
            
            if not os.path.exists(saving_path_responses):
                os.makedirs(saving_path_responses)
            if not os.path.exists(saving_path_decoder):
                os.makedirs(saving_path_decoder)
            print(model,img)
            if img == 'cue-conflict':
                kfold_file = os.path.join(os.path.abspath(os.getcwd()),"utils/style-transfer-"+kf_file)
                kf = import_kfolds(kfold_file)
                df = cross_output_trained_wog(activations_og,activations,kf,saving_path_decoder,val['epoch'],val['lr'],cue_conflict=True)
            else:
                kfold_file = os.path.join(os.path.abspath(os.getcwd()),"utils/"+kf_file)
                kf = import_kfolds(kfold_file)
                df = cross_output_trained_wog(activations_og,activations,kf,saving_path_decoder,val['epoch'],val['lr'])

            df.to_csv(saving_path_responses+f"{model}_{i}.csv")
