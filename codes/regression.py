
import torch 
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

# let x and y be series
class Dataset():
    def __init__(self, X, labels=None):
        self.X = X
        self.y = labels
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        data = torch.tensor(self.X[i, :])
            
        if self.y is not None:
            return (data, torch.tensor(self.y[i]))
        else:
            return data

class Log_Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, x):
        out = self.linear(x)
        return out
    
    def training_step(self, batch):
        data, labels = batch 
        data, labels = data.to(device), labels.to(device)
        # Generate predictions
        out = self(data.float())             
        # Calculate loss
        loss = F.cross_entropy(out, labels)   
        return loss

    def validation_step(self, batch):
        data, labels = batch 
        data, labels = data.to(device), labels.to(device)
        # Generate predictions
        out = self(data.float())            
        # Calculate loss
        loss = F.cross_entropy(out, labels)   
        return loss
    
    def testing_step(self, batch):
        data, labels = batch 
        data, labels = data.to(device), labels.to(device)
        # Generate predictions
        out = self(data.float())  
        # Apply softmax for each output row to get probabilities
        probs = F.softmax(out, dim=1)               
        return out, labels, probs

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def evaluate(model, test_loader):
    outputs_tr = torch.empty((1,16))
    labels_tr = torch.empty((1))
    probs_tr = torch.empty((1,16))

    for batch in test_loader:
        outputs, labels, probs = model.testing_step(batch) 
        outputs, labels, probs = outputs.cpu(),labels.cpu(),probs.cpu()
        outputs_tr = torch.cat((outputs_tr,outputs),axis=0)
        labels_tr = torch.cat((labels_tr,labels),axis=0)
        probs_tr = torch.cat((probs_tr,probs),axis=0)
        
    outputs_tr = outputs_tr[1:,:]
    labels_tr = labels_tr[1:]
    probs_tr = probs_tr[1:,:]
    acc = accuracy(outputs_tr,labels_tr)
    #print(acc)
    return acc, labels_tr, probs_tr

from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

def fit(epoch, lr, model, train_loader, test_loader, opt_func=SGD):
    optimizer = opt_func(model.parameters(), lr)
    scheduler = StepLR(optimizer, step_size=40, gamma=0.5)
    training_loss = []
    validation_loss = []
    
    # Training Phase 
    for epoch in range(epoch):
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
        #print(f'Epoch: {epoch}, Loss: {loss}')
        training_loss.append(loss.cpu().detach().numpy())

        for batch in test_loader:
            val_loss = model.validation_step(batch)
        validation_loss.append(val_loss.cpu().detach().numpy())

    # Testing phase
    with torch.no_grad():
        acc, labels_tr, pred_prob = evaluate(model, test_loader)
    return acc, labels_tr, pred_prob, training_loss, validation_loss


###############################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from sklearn.utils import shuffle

# let x and y be series
def data_process(x_train, y_train, x_test, y_test, batch_train = None, batch_test= None):
    train_data = Dataset(x_train,y_train)
    test_data = Dataset(x_test, y_test)

    if batch_train == None:
        batch_train = len(train_data)
    if batch_test == None:
        batch_test = len(test_data)
    # dataloaders
    trainloader = DataLoader(train_data, batch_size=batch_train)
    # do not shuffle test daat
    testloader = DataLoader(test_data, batch_size=batch_test)
    return trainloader,testloader

import os
import json
path = os.path.join(os.path.abspath(os.getcwd()),"utils/reference.json")
with open(path) as f:
    ref_dict = json.load(f)


class model_fitting():
    def __init__(self,epoch,lr):
        self.auc = 0
        # dictionary containing loss over epochs for each fold 
        self.train_loss = {}
        self.val_loss = {}
        self.epoch = epoch
        self.lr = lr

    def train_model(self,activation_file, kfolds, model_type=None):
        df = pd.read_csv(activation_file, index_col =0)
        #print(len(df.index))
        #print(df.head())
        X = df.iloc[:,:-1].to_numpy()
        Y = df.iloc[:,-1]
        le = preprocessing.LabelEncoder()
        Y = le.fit_transform(Y)

        first = True
        i = 1
        auc =[]
        for key, val in kfolds.items():
            #print(key)
            train_index = val['train']
            test_index = val['test']

            x_train = X[train_index,:]
            y_train = Y[train_index]
            x_test = X[test_index,:]
            y_test = Y[test_index]
            #print(y_test)
            x_train, y_train = shuffle(x_train,y_train,random_state=66) 

            # keep track of which image corresponds to which label
            img_labels = df.index[test_index].values.reshape(-1,1)

            model = Log_Model(x_test.shape[1],16).to(device)
            train, test = data_process(x_train, y_train, x_test, y_test)
            # get accuracy, training loss and validation loss for each fold over epochs
            acc, labels, pred_prob, train_loss, val_loss = fit(self.epoch, self.lr, model,train,test)

            self.train_loss[key] = train_loss
            self.val_loss[key] = val_loss
            auc.append(acc.detach().numpy())

            labels = le.inverse_transform(labels.numpy().astype(int)).reshape(-1,1)
            n = pred_prob.detach().numpy()
            responses = le.inverse_transform(pred_prob.argmax(1).numpy()).reshape(-1,1)
            #print(labels,responses)

            df_k = np.concatenate((img_labels, n),axis=1)
            if first == True:
                df_resp = df_k
                first = False
            else:
                df_resp = np.concatenate((df_resp, df_k))

            if model_type!=None:
                strNum = str(i)
                model_name = os.path.join(model_type + strNum.zfill(3) + ".pth")
                torch.save(model.state_dict(), model_name)
                i += 1

        columns = ["unique_img", "airplane", "bear", "bicycle", "bird", "boat", "bottle", "car", "cat", "chair", "clock", "dog", "elephant", "keyboard", "knife", "oven", "truck"]
        df_resp = pd.DataFrame(df_resp, columns=columns).sort_values(by="unique_img").set_index("unique_img")
        #print(df_resp.head())

        # overall accuracy 
        self.auc = np.nanmean(auc)

        return df_resp

    def train_model_with_originals(self,activation_file1, activation_file2, kfolds1, model_type=None,cue_conflict=False):
        df_og = pd.read_csv(activation_file1, index_col=0)
        df = pd.read_csv(activation_file2, index_col =0)
        #print(len(df.index))
        #print(df.head())
        le = preprocessing.LabelEncoder()
        X_og = df_og.iloc[:,:-1].to_numpy()
        Y_og = le.fit_transform(df_og.iloc[:,-1])
        X = df.iloc[:,:-1].to_numpy()
        Y = le.fit_transform(df.iloc[:,-1])
        #print(X_og.shape,Y_og.shape)
        first = True
        i = 1
        auc =[]
        for key, val in kfolds1.items():
            print(key)
            train_index = val['train']
            test_index = val['test']

            x_test = X[test_index,:]
            y_test = Y[test_index]

            x_train = np.concatenate((X[train_index,:],X_og),axis=0)
            y_train = np.concatenate((Y[train_index],Y_og),axis=0)
            x_train, y_train = shuffle(x_train,y_train,random_state=66) 
            # keep track of which image corresponds to which label
            img_labels = np.array(df.index[test_index]).reshape(-1,1)
            model = Log_Model(x_train.shape[1],16).to(device)

            train, test = data_process(x_train, y_train, x_test, y_test)
            # get accuracy, training loss and validation loss for each fold over epochs
            acc, labels, pred_prob, train_loss, val_loss = fit(self.epoch, self.lr, model,train,test)
            labels = le.inverse_transform(labels.numpy().astype(int)).reshape(-1,1)
            n = pred_prob.detach().numpy().astype(float)
            responses = le.inverse_transform(pred_prob.argmax(1).numpy()).reshape(-1,1)
            #print(labels,responses)
            df_k = np.concatenate((img_labels, n),axis=1)
            if first == True:
                df_resp = df_k
                first = False
            else:
                df_resp = np.concatenate((df_resp, df_k))
            if model_type!=None:
                strNum = str(i)
                model_name = os.path.join(model_type + strNum.zfill(3) + ".pth")
                torch.save(model.state_dict(), model_name)
                i += 1

        columns= ["unique_img", "airplane", "bear", "bicycle", "bird", "boat", "bottle", "car", "cat", "chair", "clock", "dog", "elephant", "keyboard", "knife", "oven", "truck"]
        df_raw = pd.DataFrame(df_resp, columns=columns)
        #print(df_raw.head())
        for i in range(1,len(columns)):
            df_raw[columns[i]] = pd.to_numeric(df_raw[columns[i]])
        df_fin = df_raw.groupby("unique_img",as_index=False).mean()
        df_fin = df_fin.sort_values(by="unique_img").set_index("unique_img")
        #print(df_fin.head())

        # overall accuracy 
        self.auc = np.nanmean(auc)

        return df_fin

    def plot_training_loss(self):
        plt.clf()
        fig = plt.figure()
        ax = fig.gca()
        for fold in self.train_loss:
            ax.plot(self.train_loss[fold],label=fold)
        ax.legend(loc='upper right')
        plt.semilogy()
        return fig

    def plot_validation_loss(self):
        plt.clf()
        fig = plt.figure()
        ax = fig.gca()
        for fold in self.val_loss:
            ax.plot(self.val_loss[fold],label=fold)
        ax.legend(loc='upper right')
        plt.semilogy()
        return fig

import os 

def get_parameter(activation_file,kf,saving_path):
    lr_list = np.arange(0.005,0.015,0.001)
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    outfile = open(os.path.join(saving_path,"results.csv"), 'w')
    outfile.writelines(f'Epoch, Learning Rate, Mean Accuracy \n')

    for epoch in range(100,300,100):
        for lr in lr_list:
            auc = []
            for i in range(5):
                this_model = model_fitting(epoch,lr)
                df = this_model.train_model(activation_file, kf)
                fig1 = this_model.plot_training_loss()
                fig1 = plt.savefig(os.path.join(saving_path,f"{epoch}_{lr}_tl.png"))
                fig2 = this_model.plot_validation_loss()
                fig2 = plt.savefig(os.path.join(saving_path,f"{epoch}_{lr}_vl.png"))
                auc.append(this_model.auc)

            print(f'Epoch: {epoch}, learning rate: {lr}, accuracy: {np.nanmean(auc)}' )
            outfile = open(os.path.join(saving_path,"results.csv"), 'a')
            outfile.writelines(f'{epoch}, {lr}, {np.nanmean(auc)}\n')
            outfile.close()
    return

# saving_path should be in the form of dir/{img type}/{model name}/models/
def cross_output(activation_file,kf,saving_path,epoch,lr):
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    this_model = model_fitting(epoch,lr)
    df = this_model.train_model(activation_file, kf, saving_path)
    fig1 = this_model.plot_training_loss()
    fig1 = plt.savefig(os.path.join(saving_path,f"{epoch}_{lr}_tl.png"))
    plt.clf()
    fig2 = this_model.plot_validation_loss()
    fig2 = plt.savefig(os.path.join(saving_path,f"{epoch}_{lr}_vl.png"))
    plt.clf()
    return df

# saving_path should be in the form of dir/{img type}/{model name}/models/
def cross_output_trained_wog(activation_og, activation_file,kf,saving_path,epoch,lr,cue_conflict=False):
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    this_model = model_fitting(epoch,lr)
    df = this_model.train_model_with_originals(activation_og, activation_file, kf, saving_path,cue_conflict)
    #fig1 = this_model.plot_training_loss()
    #fig1 = plt.savefig(os.path.join(saving_path,f"{epoch}_{lr}_tl.png"))
    #plt.clf()
    #fig2 = this_model.plot_validation_loss()
    #fig2 = plt.savefig(os.path.join(saving_path,f"{epoch}_{lr}_vl.png"))
    #plt.clf()
    return df