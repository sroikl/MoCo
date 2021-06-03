import torch
from torch import  nn
import torchvision.models as models
class MoCo(nn.Module):
    def __init__(self,m,queue):
        super(MoCo,self).__init__()

        self.keys_encoder= models.resnet50(num_classes=128)
        self.query_encoder= models.resnet50(num_classes= 128)
        self.m= m

        ''' Queue Initialization'''
        self.queue_ptr= 0
        _,self.queue_size= queue.shape
        self.queue = nn.functional.normalize(queue, dim=0)

        '''Initializing the Keys Encoder and SG'''
        for param_q, param_k in zip(self.query_encoder.parameters(), self.keys_encoder.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def forward(self, x_k,x_q,Train=True):
        ''' Augment x twice -> pass through k/q networks -> '''

        q= self.query_encoder(x_q)
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            if Train:
                self._momentum_contrast_update()

            k= self.keys_encoder(x_k)
            k = nn.functional.normalize(k, dim=1)
            k= k.detach()

        N,C= k.shape#shape of input data (NxC)
        _,K= self.queue.shape

        #Create Logits
        # l_pos= torch.bmm(q.view(N,1,C),k.view(N,C,1)).squeeze(-1) #dim (Nx1)
        # l_neg= torch.mm(q.view(N,C),self.queue.clone().detach().view(C,K)) #dim (NxK)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)

        if Train:
            self._UpdateQueue(keys=k)

        return logits,k

    def _momentum_contrast_update(self):
        for theta_q, theta_k in zip(self.query_encoder.parameters(),
                                    self.keys_encoder.parameters()):

            theta_k.data = theta_k.data * self.m + theta_q.data * (1. - self.m)


    def _UpdateQueue(self,keys):
        N, C = keys.shape
        ''' ========  Enqueue/Dequeue  ======== '''
        self.queue = torch.cat((self.queue, keys.t()), dim=1)
        self.queue = self.queue[:, N:]

class DownStreamTaskModel(nn.Module):
    def __init__(self,encoder):
        super(DownStreamTaskModel,self).__init__()

        self.encoder= encoder
        self.fc= nn.Sequential(nn.Linear(),nn.ReLU(),nn.Dropout())

    def forward(self,batch):
        return self.fc(self.encoder(batch))
