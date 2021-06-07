import torch
from torch import nn

class MoCo(nn.Module):

    def __init__(self, backbone,output_dim_list):
        super(MoCo, self).__init__()

        self.fc_length = backbone.fc.in_features # Extract output fc layer depth
        self.encoder= backbone # Define base encoder as the backbone network
        self.linear = LinearBlock(input_dim=self.fc_length,output_dim_list=output_dim_list)
        self.encoder.fc = nn.Sequential(self.linear) # Construct a new output block for the base encoder


    def forward(self, x):

        y = self.encoder(x)

        return y


class LinearBlock(nn.Module):
    def __init__(self,input_dim,output_dim_list):
        super(LinearBlock,self).__init__()

        mum_layers= len(output_dim_list)

        layers=[]
        for i,output_dim in enumerate(output_dim_list):
            if i == mum_layers-1:
                layers.append(nn.Linear(in_features=input_dim,out_features=output_dim))
            else:
                layers.append(nn.Linear(in_features=input_dim, out_features=output_dim), nn.ReLU())

            input_dim=output_dim

        self.fc= nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


class DownStreamTaskModel(nn.Module):
    def __init__(self,encoder,InputDim,OutputDim_List):
        super(DownStreamTaskModel,self).__init__()

        self.encoder= encoder
        self.fc= LinearBlock(input_dim=InputDim,output_dim_list=OutputDim_List)

    def forward(self,x):
        features= self.encoder(x)
        features = nn.functional.normalize(features, p=2, dim=1)
        features= features.view(-1).copy()
        return self.fc(features)
