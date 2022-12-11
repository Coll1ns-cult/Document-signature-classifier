import torch
import torch.nn as nn
from tqdm import tqdm

def evaluate_one_episode(x:torch.tensor,
                        y:torch.tensor,
                        n_shot:int,
                        q:int, 
                        model:nn.Module,
                        device: torch.device):
    '''Function to evaluate one episode
        -----------------
        Args: 
        x: feature tensor, shape: (n_shot*k_way+q, embedding dimension)
        y: label vectors, shape: (n_shot*k_way+q)
        n_shot: #shots (i.e #support samples)
        q: #qeury samples
        ------
        return: #correctly found labels, lenght of query_labels (or q) '''
    k_way = (y.shape[0] - q)//n_shot
    query_labels = y[n_shot*k_way:]
    return (torch.max(model(x, n_shot, k_way, q).detach().data,1,
            )[1]== query_labels.to(device)).sum().item(), q



def evaluate(data_loader
            model: nn.Module, ):
    total_predictions = 0
    correct_predictions = 0

    # eval mode affects the behaviour of some layers (such as batch normalization or dropout)
    # no_grad() tells torch not to keep in memory the whole computational graph (it's more lightweight this way)
    model.eval()
    # with torch.no_grad()
    
    for episode_index, (x, y) in tqdm(enumerate(data_loader), total=len(data_loader)):
        y = y.squeeze(0).to(device)
        # y_sample = y.clone().to(device)
        # print(new_label(y_sample), "this one")
        # for i in range(k_way):
        #   y[i*n_shot:(i+1)*n_shot] = i
        #   y[n_shot*k_way+i*q:n_shot*k_way+(i+1)*q]=i
        # print(y)
        y = not_sorted_label(y)
        correct, total = evaluate_one_episode(
            x, y, n_shot, q
        )

        total_predictions += total
        correct_predictions += correct

    print(
        f"Model tested on {len(data_loader)} tasks. Accuracy: {(100 * correct_predictions/total_predictions):.2f}%"
    )
