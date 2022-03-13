# getting feature activations

import torch, torchvision
from torch.nn.modules.pooling import AdaptiveAvgPool2d
from PIL import Image
from torchvision import transforms
import pandas as pd

def get_alexnet(inter=True):
    model = torchvision.models.alexnet(pretrained=True)
    model.eval()

    if inter==False:
        return model

    # alexnet without final layer
    new_model = torch.nn.Sequential(*(list(model.features.children())+[model.avgpool]+[torch.nn.Flatten()]+list(model.classifier.children())[:-1]))
    return new_model

def get_resnet18(inter=True):
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()

    if inter==False:
        return model

    new_model = torch.nn.Sequential(*(list(model.children())[:-1]+[torch.nn.Flatten()]))
    return new_model

def get_resnet34(inter=True):
    model = torchvision.models.resnet34(pretrained=True)
    model.eval()

    if inter==False:
        return model

    new_model = torch.nn.Sequential(*(list(model.children())[:-1]+[torch.nn.Flatten()]))
    return new_model

def get_resnet50(inter=True):
    model = torchvision.models.resnet50(pretrained=True)
    model.eval()

    if inter==False:
        return model
    
    new_model = torch.nn.Sequential(*(list(model.children())[:-1]+[torch.nn.Flatten()]))
    return new_model

def get_resnet101(inter=True):
    model = torchvision.models.resnet101(pretrained=True)
    model.eval()

    if inter==False:
        return model

    new_model = torch.nn.Sequential(*(list(model.children())[:-1]+[torch.nn.Flatten()]))
    return new_model

def get_vgg11(inter=True):
    model = torchvision.models.vgg11(pretrained=True)
    model.eval()

    if inter==False:
        return model

    new_model = torch.nn.Sequential(*(list(model.features.children())+[model.avgpool]+[torch.nn.Flatten()]+list(model.classifier.children())[:-1]))
    return new_model

def get_vgg13(inter=True):
    model = torchvision.models.vgg13(pretrained=True)
    model.eval()

    if inter==False:
        return model

    new_model = torch.nn.Sequential(*(list(model.features.children())+[model.avgpool]+[torch.nn.Flatten()]+list(model.classifier.children())[:-1]))
    return new_model

def get_vgg16(inter=True):
    model = torchvision.models.vgg16(pretrained=True)
    model.eval()

    if inter==False:
        return model

    new_model = torch.nn.Sequential(*(list(model.features.children())+[model.avgpool]+[torch.nn.Flatten()]+list(model.classifier.children())[:-1]))
    return new_model

def get_vgg19(inter=True):
    model = torchvision.models.vgg19(pretrained=True)
    model.eval()

    if inter==False:
        return model

    new_model = torch.nn.Sequential(*(list(model.features.children())+[model.avgpool]+[torch.nn.Flatten()]+list(model.classifier.children())[:-1]))
    return new_model

def get_densenet121(inter=True):
    model = torchvision.models.densenet121(pretrained=True)
    model.eval()

    if inter==False:
        return model

    new_model = torch.nn.Sequential(*(list(model.features.children())+[torch.nn.ReLU()]+[torch.nn.AdaptiveAvgPool2d((1,1))]+[torch.nn.Flatten()]))
    return new_model

def get_densenet169(inter=True):
    model = torchvision.models.densenet169(pretrained=True)
    model.eval()

    if inter==False:
        return model

    new_model = torch.nn.Sequential(*(list(model.features.children())+[torch.nn.ReLU()]+[torch.nn.AdaptiveAvgPool2d((1,1))]+[torch.nn.Flatten()]))
    return new_model

######################
# get vonenet models, first pull vonenet resposiitory from github
#import vonenet
#def get_vonenet_resnet50(inter=True):
#    model = vonenet.get_model(model_arch='resnet50',pretrained=True)
#    model.eval()
#    
#    if inter==False:
#        return model
#
#    new_model = torch.nn.Sequential(*(list(list(model.children())[0])[:-1]+list(list(list(model.children())[0])[-1].children())[:-1]+[torch.nn.Flatten()]))
#    return new_model
#
#def get_vonenet_cornets(inter=True):
#    model = vonenet.get_model(model_arch='cornets',pretrained=True)
#    model.eval()
#    
#    if inter==False:
#        return model
#
#    new_model = torch.nn.Sequential(*(list(list(model.children())[0])[:-1]+list(list(list(model.children())[0])[-1].children())[:-1]+[torch.nn.AdaptiveAvgPool2d((1,1))]+[torch.nn.Flatten()]))
#    return new_model

# preprocessor
preprocess = transforms.Compose([
        transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# vonenet mean = 0.5, std = 0.5

preprocess_vone = transforms.Compose([
        transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])

# retreive images from local directory 
import os
def image_process(path,model_name):
    print(path)
    filenames = []
    first = True
    for root, dirs, files in os.walk(path):
        #print(dirs)
        for dir in dirs:
            for subroot, subdirs, subfiles in os.walk(os.path.join(root,dir)):
                for subfile in subfiles:
                    if subfile.endswith('.png'):
                        #print('processing: ', os.path.join(subroot,subfile))
                        filenames.append(subfile)
                        input_image = Image.open(os.path.join(subroot,subfile))
                        if model_name == 'vonenet_resnet50' or model_name == 'vonenet_cornets':
                            input_tensor = preprocess_vone(input_image)
                        else:
                            input_tensor = preprocess(input_image)
                        input_batch = input_tensor.unsqueeze(0)
                        if first == True:
                            first = False
                            input = input_batch
                        else:
                            input = torch.cat((input,input_batch))
    return filenames, input

from regression import Dataset
from torch.utils.data import DataLoader
def get_activations(path,model_name):
    fnc_name = f'get_{model_name}()'
    model = eval(fnc_name)    
    filenames, input = image_process(path,model_name)
    print(input.shape)

    dataloader = DataLoader(input,50)

    have_output = False
    for batch in dataloader:
        print("Batch")
        batch_input = batch
        batch_output = model(batch_input)
        print(batch_output.shape)
        if have_output == False:
            output = batch_output
            have_output = True
        else:
            output = torch.cat((output,batch_output))
    print(output.shape)
    return filenames, output

# save file as csv
def save_activations(filenames, output):
    n = output.detach().numpy().squeeze()
    labels = get_labels(filenames, output)
    df = pd.DataFrame(n,index=filenames)
    df['category'] = labels
    df = df.sort_index()
    return df 

# get labels from filenames
def get_labels(filenames,output):
    category = ["airplane", "bear", "bicycle", "bird", "boat", "bottle", "car", "cat", "chair", "clock", "dog", "elephant", "keyboard", "knife", "oven", "truck"]
    labels = []
    for img in filenames:
        name = img.split("-")[0]
        for c in category:
            if name.find(c) != -1:
                labels.append(c)
    return labels


