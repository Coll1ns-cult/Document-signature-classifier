import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
import torch.optim 
import random
import numpy as np
import time
import tqdm
from time import sleep


def ohe(y:torch.tensor):
#one hot encoding of labels
    one_hot_y = torch.zeros(y.shape, k)
    values = []
    for i, label in enumerate(y):
        if label not in values:
            values.append(label)
            k = len(values)
            one_hot_y[i, k] = 1
        else:
            k = values.index(label)
            one_hot_y[i, k] = 1
    return one_hot_y

def train(model: Module,
          optimizer: torch.optim.Optimizer,
          train_dataloader:torch.utils.data.Dataloader,
          val_dataloader:torch.utils.data.Dataloader,
        #   N_tr: int ,
          n: int ,
          k: int ,
          q: int,
          device:torch.cuda.device,
          epochs: int):
    '''1. find Dataset #number of classes
       2.fill random sampler function from 
        3. fill examples_per_class
        4. extract examples per_class'''
    criterion = nn.NLLLoss()
    model.train()

    loss_total = 0
    accuracy = []
    for epoch in range(epochs):
        with tqdm(train_dataloader, unit="support") as tepoch:
            for (x, label) in tepoch:
                '''shape of x: (batch, n*k + q*k, channels, height, width)'''

                x.to(device)
                label.to(device)

                one_hot_y = ohe(label) #one_hot encoding
                y_S = one_hot_y[:n*k] #getting labels of support set
                y = one_hot_y[q*k:] #getting labels of query set note that this one_hot encoded y
                attention = model(x) #outputting attention
                y_hat = torch.matmul(y_S, attention).T #getting predicted labels for query set

                optimizer.zero_grad()
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()
                pred = torch.zeros_like(y_hat)
                pred = torch.fill_(1, torch.argmax(y_hat, dim = 1))
                loss_total += loss.item()
                acc = (pred == y).sum().item()/y.shape[0]
                accuracy.append(acc)


                tepoch.set_postfix(loss=loss_total, accuracy=100. * acc)
                # sleep(0.1)

            # if batch_idx % 50 == 0:
            #     print(f'Epoch : {epoch} || {batch_idx}/{len(C)} || \
            #     loss : {loss.item():.3f}, accuracy : {accuracy * 100:.3f})

            

        model.eval()
        valid_loss_total = 0
        valid_accuracy = []
        with torch.no_grad():
            with tqdm(train_dataloader, unit="support") as tepoch:
                for (x, label) in val_dataloader:

                    x = x.to(device)
                    label = label.to(device)

                    one_hot_y = ohe(label)
                    y_S = one_hot_y[:n*k]
                    y = one_hot_y[q*k:]
                    attention = model(x)
                    y_hat = torch.matmul(y_S, attention).T

                    optimizer.zero_grad()
                    loss = criterion(y_hat, y)
                    pred = torch.zeros_like(y_hat)
                    pred = torch.fill_(1, torch.argmax(y_hat, dim = 1))
                    valid_loss_total += loss.item()
                    valid_acc = (pred == y).sum().item()/y.shape[0] #write accuracy function for this, next time add F1 scores for this
                    valid_accuracy.append(valid_acc)

                    tepoch.set_postfix(loss=valid_loss_total, accuracy=100. * valid_acc)





