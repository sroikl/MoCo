import torch
from torch import  nn

class MoCo(nn.Module):
    def __init__(self,backbone):
        super(self,MoCo).__init__()

        self.keys_encoder= backbone #todo: change backbone
        self.query_encoder= backbone
        self.classifier= nn.Sequential() #todo: implement MLP Head

    def forward(self, x,queue):
        ''' Augment x twice -> pass through k/q networks -> '''


        # randomly augmented version of x
        x_q= augment(x) #todo: implement augment function
        x_k= augment(x)

        #forward pass through backbone
        k= self.keys_encoder(x_k)
        q= self.query_encoder(x_q)

        N,C= k.shape #shape of input data (NxC)
        _,K= queue.shape

        #Create Logits
        l_pos= torch.bmm(x_q.view(N,1,C),k.view(N,C,1)) #dim (Nx1)
        l_neg= torch.mm(x_q.view(q.view(N,C),queue.view(C,K))) #dim (NxK)
        logits= torch.cat((l_pos,l_neg),dim=1) #dim (Nx(K+1))

        return logits


