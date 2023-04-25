import torch
torch.manual_seed(0)

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader,Dataset
from torch.utils.tensorboard import SummaryWriter

from torchvision.models import resnet18
import matplotlib.pyplot as plt
from datetime import datetime
#import torchvision.datasets.ImageFolder 
import numpy as np
import os
#custom import
from models.model_classes.convNet import *


from models.model_classes.DNN import *
from models.model_classes.RNN import *
from const import *
from test import report_accuracies
from utils import *


# Device configuration
save=1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
import pandas as pd
from torchvision.io import read_image
def write_tensorboard_acc(writer,model,train_loader,test_loader,epoch,start_from_ep):        
    train_acc,test_acc=report_accuracies(model,train_loader,test_loader,print_result=0)
    train_acc/=100.0
    test_acc/=100.0
    print("writing tensorboard")
    writer.add_scalar("training accuracy",train_acc,start_from_ep+epoch)
    writer.add_scalar("testing accuracy",test_acc,start_from_ep+epoch)

def train(model,tf_board_dir="CNN",writer_name="",num_epochs=500,learning_rate=0.00001, tensor_board_log_step=5,start_from_ep=0):
#    MODEL_CHECKPOINT
    log_path=os.path.join("tf_board",tf_board_dir)
    writer=SummaryWriter(f"{log_path}/{writer_name}")
    print(f"lr={learning_rate}")
    print(f"batch size={batch_size}")
    loss_arr=[]
    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    n_total_steps = len(train_loader)
    with open(LOSS_LOG_PATH, 'a') as f:
        #f.writelines(f"{model.model_name}_ep{num_epochs}\n")
        f.writelines(f"training time:\n")
        f.writelines(datetime.now().strftime("%Y%m%d_%H%M%S"))
        f.write('\n')

        print(len(train_loader))
        for epoch in range(num_epochs):
            running_loss=0.0
            running_correct=0
            for i, (images, labels) in enumerate(train_loader):#loop for one batch
            
                images = images.to(device)
                labels = labels.to(device)
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss+=loss.item()
                _, predicted = torch.max(outputs.data, 1)
                running_correct+=(predicted==labels).sum().item()
                
                
                if (i+1) % 5 == 0:# print loss
                    print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            if (epoch+1)%tensor_board_log_step==0:
                write_tensorboard_acc(writer,model,train_loader,test_loader,epoch,start_from_ep)      
                #print('wr') 
            if ((epoch+1)%50==0 and save):
                print("saving model")
                torch.save(model.state_dict(), './cnn.pth')
        print('Finished Training')
        f.write("Finished Training")
        training_name=model.model_name+"_"+get_datetime()
        PATH=os.path.join(TRAINED_MODELS_PATH,training_name)
        #torch.save(model.state_dict(), PATH)
        torch.save(model.state_dict(), './cnn.pth')
    report_accuracies(model,train_loader,test_loader,batch_size=batch_size,logFile=ACC_LOG_PATH)
#Load model and train
def train_from_load(model_object,modelPath,num_epochs,learning_rate):
    
   
    #MODEL_PATH=os.path.join("trained_models","1_1_convNet2_ep=250_lr_0001.pth")
    model.load_state_dict(torch.load(modelPath,map_location=torch.device(device)),strict=False)
    model.eval()
    train(model,num_epochs=num_epochs,learning_rate=learning_rate)
    #train(model,writer_name="lr= "+str(lr),num_epochs=200,learning_rate=lr,tensor_board_log_step=10)
def train_previous(model_object,num_epochs,learning_rate):
    model.load_state_dict(torch.load('./cnn.pth',map_location=torch.device(device)),strict=False)
    model.eval()
    train(model,num_epochs=num_epochs,learning_rate=learning_rate)

if __name__=='__main__':
    print(device)
   
    train_dataset,test_dataset,train_loader,test_loader=getDatasetDataloader()
    num_classes = len(train_dataset.classes)
    print(f"number of classes: {num_classes}")
    
    lrs=[0.0005]#List of learning rate
    for lr in lrs:
        model=OptimConvNet2(output_size=num_classes)#Convolution Neural Network (CNN)
        print(f"using model: {model.model_name}")

        print(model.eval())
        
        
        #model.load_state_dict(torch.load('./cnn.pth',map_location=torch.device(device)),strict=False)
        #model.eval()
        #train(model,num_epochs=200,learning_rate=learning_rate)

        train(model,tf_board_dir="CNN",writer_name="lr= "+str(lr)
              ,num_epochs=10,learning_rate=lr,tensor_board_log_step=5,start_from_ep=0)
        model=None
        #train_from_load(model,"cnn.pth",50,0.0005)
    #train(model,100,0.001)
