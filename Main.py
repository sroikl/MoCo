from DataModule import DataModule
from Model import MoCo,DownStreamTaskModel
import torch
from Trainer import MoCoTrainer
import torchvision.models as models


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f'Working with {device}, number of GPUs: {torch.cuda.device_count()}')

def model_pipeline():
    '''Phase A - train Momentum Encoder'''

    #HyperParameters
    backbone= models.resnet50(num_classes= 128)

    tau= 0.07
    momentum= 0.999
    lr= 1e-4
    queue_size= 1028
    batch_size= 64
    epochs= 200

    image_size = 224
    ks = (int(0.1 * image_size) // 2) * 2 + 1  # should be odd
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    #define initial queue
    queue= torch.randn(128,queue_size).to(device=device) #TODO:update


    #main model training
    dl_train,dl_val,transforms= DataModule(batch_size= batch_size,ks=ks,imagenet_stats=__imagenet_stats)
    moco_model= MoCo(backbone= backbone,m=momentum,queue=queue).to(device=device)
    optimizer= torch.optim.SGD(params=moco_model.parameters(),lr=lr,weight_decay=1e-4,momentum=0.9)
    lr_schedualer = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120, 160], gamma=0.1)
    loss_fn= torch.nn.CrossEntropyLoss()

    trainer= MoCoTrainer(model= moco_model, loss_fn=loss_fn,optimizer=optimizer,scheduler=lr_schedualer,tau=tau,flg=True,device=device)

    trainer.fit(dl_train= dl_train,dl_val= dl_val, epochs= epochs)

    '''Phase B - train Downstream task'''


if __name__ == '__main__':

    model_pipeline()