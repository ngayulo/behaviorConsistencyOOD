
from features import *
import os 

image_paths = {
#    'original': "input/original/",
    'edges': "input/edges/",
    'silhouette': "input/silhouettes/",
    'cue-conflict': "input/style-transfer/",
    'grayscale': "input/grayscale/"
}

exp_code = input("Enter experiment code: ")

#'resnet18', 'resnet34', 'resnet50', 'resnet101','vgg11','vgg13','vgg16','vgg19', 'densenet121','densenet169'
model_names = ['alexnet','resnet18', 'resnet34', 'resnet50', 'resnet101','vgg11','vgg13','vgg16','vgg19', 'densenet121','densenet169']

def features_extraction(model,img_type):
    path = os.path.join(os.path.abspath(os.getcwd()),image_paths[img_type])

    saving_path = os.path.join(os.path.abspath(os.getcwd()),f'{exp_code}/{img_type}/{model}')
    outfile = os.path.join(saving_path,'{}_feature_activation.csv'.format(model))

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
        
    out = os.open(outfile,os.O_CREAT)
    os.close(out)
    filenames, output = get_activations(path,model)
    df = save_activations(filenames, output)
    print(df.head())
    df.to_csv(outfile)
    print("Done")
    return 

def main():
    for model_name in model_names:
        for img_type in image_paths:
            features_extraction(model_name,img_type)
    return

if __name__ == '__main__':
    main()
