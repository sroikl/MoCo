from DataModule import DataModule
from Model import MoCo,DownStreamTaskModel
import torch
from Trainer import MoCoTrainer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Working with {device}, number of GPUs: {torch.cuda.device_count()}')

def model_pipeline():
    '''Phase A - train Momentum Encoder'''

    #HyperParameters
    backbone=
    tau= 0.07
    momentum= 0.999
    lr= 1e-3
    queue_size= 4096
    batch_size= 256
    epochs= 100
    #Upload BackBone

    #define initial queue

    queue= torch.zeros(1024,queue_size) #TODO:update
    #main model training
    dl_train,dl_val,transforms= DataModule()
    moco_model= MoCo(backbone= backbone,transforms=transforms)
    optimizer= torch.optim.SGD(params=moco_model.parameters(),lr=lr,weight_decay=1e-4)
    lr_schedualer= ; #todo:define
    loss_fn= torch.nn.CrossEntropyLoss()

    trainer= MoCoTrainer(model= moco_model, loss_fn=loss_fn,optimizer=optimizer,scheduler=lr_schedualer,tau=tau,
                        queue=queue,momento=momentum,flg=,device=device)

    trainer.fit(dl_train= dl_train,dl_val= dl_val, epochs= epochs)

    '''Phase B - train Downstream task'''


if __name__ == '__main__':

    model_pipeline()