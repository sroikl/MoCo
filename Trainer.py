import torch

class MoCoTrainer():
    def __init__(self,dl_train,dl_val,model,optimizer,scheduler,epochs,device):
        super(self,MoCoTrainer).__init__()

        self.dl_train= dl_train
        self.dl_val= dl_val
        self.model= model
        self.optimizer= optimizer
        self.scheduler= scheduler
        self.epochs= epochs
        self.device= device\


    def fit(self,):

        for epoch in range(self.epochs):

            '''Train Part'''
            for batch in self.dl_train:
                loss= self.train_batch(batch)

            '''Validation'''
            for batch in self.dl_val:


    def train_batch(self,batch):
        batch= batch.to(self.device)

        #Forward Pass
        logits=
    def val_batch(sel,batch):
        pass