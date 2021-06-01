import torch
from torch import  nn

class MoCo(nn.Module):
    def __init__(self,backbone,m,queue):
        super(MoCo,self).__init__()

        self.keys_encoder= backbone #todo: change backbone
        self.query_encoder= backbone
        self.m= m
        self.queue= queue

    def forward(self, x_k,x_q):
        ''' Augment x twice -> pass through k/q networks -> '''

        with torch.no_grad():
            k= self.keys_encoder(x_k)
            k = nn.functional.normalize(k, dim=1)

        k= k.detach()


        q= self.query_encoder(x_q)
        q = nn.functional.normalize(q, dim=1)


        N,C= k.shape#shape of input data (NxC)
        _,K= self.queue.shape

        #Create Logits
        l_pos= torch.bmm(q.view(N,1,C),k.view(N,C,1)).squeeze(-1) #dim (Nx1)
        l_neg= torch.mm(q.view(N,C),self.queue.clone().view(C,K)) #dim (NxK)

        logits= torch.cat((l_pos,l_neg),dim=1) #dim (Nx(K+1))

        self._momentum_contrast_update()
        self._UpdateQueue(keys=k)
        return logits,k

    def _momentum_contrast_update(self):
        for theta_q, theta_k in zip(self.query_encoder.parameters(),
                                    self.keys_encoder.parameters()):  # TODO:vectorize

            c = theta_k.data.sum()
            # print(c)
            a = (1. - self.m) * theta_q.data
            b = self.m * theta_k.data
            # print(a.sum())
            # print(b.sum())
            theta_k.data = theta_k.data * self.m + theta_q.data * (1. - self.m)
            # print(theta_k.data.sum())
            # if c!= a.sum()+b.sum():
            #     p=[]

    def _UpdateQueue(self,keys):
        N, C = keys.shape
        ''' ========  Enqueue/Dequeue  ======== '''
        self.queue = torch.cat((self.queue, keys.t()), dim=1)  # TODO:DEBUG THE FUCK OUT OF THIS PART
        self.queue = self.queue[:, N:]

class DownStreamTaskModel(nn.Module):
    def __init__(self,encoder):
        super(DownStreamTaskModel,self).__init__()

        self.encoder= encoder
        self.fc= nn.Sequential(nn.Linear(),nn.ReLU(),nn.Dropout())

    def forward(self,batch):
        return self.fc(self.encoder(batch))
