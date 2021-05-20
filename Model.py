import torch
from torch import  nn

class MoCo(nn.Module):
    def __init__(self,backbone,transforms):
        super(MoCo,self).__init__()

        self.keys_encoder= backbone #todo: change backbone
        self.keys_encoder.detach() #Detach Gradients - dont want to optimize
        self.query_encoder= backbone
        self.classifier= nn.Sequential() #todo: implement MLP Head
        self.transforms= transforms

    def forward(self, x,queue):
        ''' Augment x twice -> pass through k/q networks -> '''


        # randomly augmented version of x
        x_q= self.transforms(x) #todo: implement augment function
        x_k= self.transforms(x)

        #forward pass through backbone
        k= self.keys_encoder(x_k)
        q= self.query_encoder(x_q)

        N,C= k.shape() #shape of input data (NxC)
        _,K= queue.shape()

        #Create Logits
        l_pos= torch.bmm(q.view(N,1,C),k.view(N,C,1)) #dim (Nx1)
        l_neg= torch.mm(q.view(q.view(N,C),queue.view(C,K))) #dim (NxK)
        logits= torch.cat((l_pos,l_neg),dim=1) #dim (Nx(K+1))


        return logits,k


class DownStreamTaskModel(nn.Module):
    def __init__(self,encoder):
        super(DownStreamTaskModel,self).__init__()

        self.encoder= encoder
        self.fc= nn.Sequential(nn.Linear(),nn.ReLU(),nn.Dropout())

    def forward(self,batch):
        return self.fc(self.encoder(batch))
