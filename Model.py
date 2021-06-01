import torch
from torch import  nn

class MoCo(nn.Module):
    def __init__(self,backbone):
        super(MoCo,self).__init__()

        self.keys_encoder= backbone #todo: change backbone
        self.query_encoder= backbone

        # for theta_q, theta_k in zip(self.query_encoder.parameters(),self.keys_encoder.parameters()):
        #     theta_k.data.copy_(theta_q.data)
        #     theta_k.requires_grad = False



    def forward(self, x_k,x_q,queue):
        ''' Augment x twice -> pass through k/q networks -> '''

        with torch.no_grad():
            k= self.keys_encoder(x_k)
            k = nn.functional.normalize(k, dim=1)

        k= k.detach()


        q= self.query_encoder(x_q)
        q = nn.functional.normalize(q, dim=1)


        N,C= k.shape#shape of input data (NxC)
        _,K= queue.shape

        #Create Logits
        l_pos= torch.bmm(q.view(N,1,C),k.view(N,C,1)).squeeze(-1) #dim (Nx1)
        l_neg= torch.mm(q.view(N,C),queue.clone().view(C,K)) #dim (NxK)

        logits= torch.cat((l_pos,l_neg),dim=1) #dim (Nx(K+1))

        return logits,k


class DownStreamTaskModel(nn.Module):
    def __init__(self,encoder):
        super(DownStreamTaskModel,self).__init__()

        self.encoder= encoder
        self.fc= nn.Sequential(nn.Linear(),nn.ReLU(),nn.Dropout())

    def forward(self,batch):
        return self.fc(self.encoder(batch))
