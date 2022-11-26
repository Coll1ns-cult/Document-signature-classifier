import torch 
import torch.nn as nn
import torch.nn.functional as F



class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        '''Encoder network to encode the various inputs to embedding space
        Arguments
        ---------
        input_channels: number of channels of the input image
        hidden_channels: number of channels of hidden features'''
        super(Encoder).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.main = nn.Sequential(
            nn.Conv2d(self.input_channels, self.hidden_channels, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d,
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d,
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d,
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d,
            nn.AdaptiveMaxPool2d((1,1)),
            nn.Flatten()
        )

    def forward(self, x):
        return self.main(x)

class AttLSTM(nn.Module):
    def __init__(self, K, input_size, hidden_size):
        '''attention LSTM with skip connections
        Arguments
        ---------
        K: number of procesing steps
        input_size: size of input
        hidden_size: size of hidden features'''
        super(AttLSTM).__init__()
        self.processing = K #number of times to run lstm cells ( number of lstm cells basically)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size)
        self.softmax = nn.Softmax(dim = 1)
    def forward(self, f_x, g_S):
        h = f_x
        c = torch.zeros(f_x.shape[0], f_x.shape[1]).cuda() #putting our tensor to device
        for _ in range(self.processing):
            a = self.softmax(h@g_S.T) #attention
            r = torch.sum(g_S*a, dim = 0) #summation over sequential data
            concat = torch.cat((h, r), dim = 1) #concatination
            h, c  = self.lstm(f_x, concat, c) #output of LSTM
            h = h + f_x #skip connection
        return h

class Matching(nn.Module):# change attention LSTM to transformer.(i.e add transformer option) 
    #change bidirectional lstm to one with skip connection.
    def __init__(self, K, input_size, hidden_size, layers, n_shot, k_way, q, is_full_context_embedding = False):
        super(Matching).__init__()
        self.layers = layers
        self.processing = K
        self.n_shot = n_shot
        self.k_way  = k_way
        self.q = q
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.is_full_context_embedding = is_full_context_embedding
        self.mapping = Encoder(1, self.input_size)
        self.biLSTM = nn.LSTM(self.input_size, self.layers, self.hidden_size, bidirectional = True) #set other arguments as well
        self.attLSTM = AttLSTM(K, self.input_size, self.hidden_size) #set other arguments as well 
        # self.cos = nn.CosineSimilarity(dim=2)
        self.softmax = nn.Softmax(dim = 1)
    def cosine_similarity_matrix(self, x:torch.tensor,
                                    y:torch.tensor,
                                    eps:torch.tensor):
            '''Cosine similarity for matrices, whose columns are same
            x: shape (q, e): query set
            y: shape (s, e): support set
            eps: for numerical stabilizity
            output: (q,s): columns of this matrice correspond to 
            attention vector of each sample from query'''
            matrix_product = torch.dot(x, y.T)
            l2_norm_x = torch.norm(x, p = 2, dim = 1)
            l2_norm_y = torch.norm(y, p = 2, dim = 1)
            denominator = torch.maximum(torch.dot(l2_norm_x, l2_norm_y.T), eps)
            return matrix_product/denominator
    def forward(self, x):

        if self.is_full_context_embedding:
            x  = self.mapping(x)
            support_set = x[self.n_shot*self.k_way]
            s,h,w = support_set.shape
            support_set = support_set.view(s, 1, h, w)
            query_set = x[:self.q*self.k_way]

            g_S = self.biLSTM(support_set)
            support_set.view(s, h, w)
            f_x = self.attLSTM(query_set, support_set)

            return self.softmax(self.cosine_similarity_matrix(f_x, g_S))
        else:
            x = self.mapping(x)
            support_set = x[self.n_shot*self.k_way]

            query_set = x[:self.q*self.k_way]
            return self.softmax(self.cosine_similarity_matrix(query_set, support_set))
