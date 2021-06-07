import torch
from torch import nn

class MoCo(nn.Module):

    def __init__(self, backbone):
        super(MoCo, self).__init__()

        self.fc_length = backbone.fc.in_features # Extract output fc layer depth
        self.encoder= backbone # Define base encoder as the backbone network
        self.linear = nn.Linear(self.fc_length,self.fc_length) # Define a linear layer in accordance with original output depth
        self.encoder.fc = nn.Sequential(self.linear,nn.ReLU(),backbone.fc) # Construct a new output block for the base encoder

        # ''' Queue Initialization'''
        # self.queue_ptr= 0
        # _,self.queue_size= queue.shape
        # self.queue = nn.functional.normalize(queue, dim=0)

    def forward(self, x):

        y = self.encoder(x)

        return y

# class DownStreamTaskModel(nn.Module):
#     def __init__(self,encoder):
#         super(DownStreamTaskModel,self).__init__()
#
#         self.encoder= encoder
#         self.fc= nn.Sequential(nn.Linear(),nn.ReLU(),nn.Dropout())
#
#     def forward(self,batch):
#         return self.fc(self.encoder(batch))
