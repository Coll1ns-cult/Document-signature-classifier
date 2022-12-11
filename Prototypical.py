import torch
import torch.nn as nn
import torch.nn.functional as F

#import encoder as well
#can proto network function as well if you want::: 
class Prototypical(nn.Module):
    def __init__(self, ) -> None:
        super(Prototypical, self).__init__()
     #here y_support is already made label:
    def forward(feature, n_shot, k_way, q):
        # x = x.squeeze(0).to(device)
        # embedded_x = self.backbone(x)
        # print(embedded_x.shape)
        support_set = feature[:k_way*n_shot] #shape of n*k, embedding dim 
        q_set = feature[k_way*n_shot:] #shape of q*k, embedding dim
        mean_support = torch.cat([torch.mean(support_set[i*n_shot:(i+1)*n_shot], dim = 0).unsqueeze(0) for i in range(k_way)]) #now we have 
        # print(q_set.shape, mean_support.shape)
        l2_distance = torch.cdist(q_set, mean_support)
        # print(l2_distance.shape, 'shape of l2 distance matrix')
        return -l2_distance


