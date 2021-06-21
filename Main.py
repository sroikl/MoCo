from DataModule import DataModule
from Model import MoCo,DownStreamTaskModel #, DownStreamTaskModel
import torch
from MoCoTrainer import MoCoTrainer, DictionaryQueue
from DST_Trainer import DST_Trainer
import torchvision.models as models
import os


def TrainMocoContrastive(ConfigFile):
    '''Phase A - train Momentum Encoder'''

    # Construct and Initialize dictionary queue:
    Queue = DictionaryQueue(ConfigFile['output_size'], ConfigFile['queue_size'],ConfigFile['device'])

    # Construct dataloaders:
    dl_train, dl_val, transforms = DataModule(batch_size=ConfigFile['batch_size'], ks=ConfigFile['Gauss_kernel'],
                                              imagenet_stats=ConfigFile['imagenet_stats'])

    '''Initializing the Keys Encoder and SG with matching weights'''
    # Construct base encoder models:
    Qencoder = MoCo(backbone=models.resnet50(num_classes= ConfigFile['output_size']),
                    output_dim_list= ConfigFile['output_dim_list']).to(device= ConfigFile['device'])

    Kencoder = MoCo(backbone=models.resnet50(num_classes= ConfigFile['output_size']),
                    output_dim_list= ConfigFile['output_dim_list']).to(device= ConfigFile['device'])
    # Match the weights:
    for param_q, param_k in zip(Qencoder.parameters(), Kencoder.parameters()):
        param_k.data.copy_(param_q.data)

    # Define the optimization process:
    optimizer = torch.optim.Adam(params=Qencoder.parameters(), lr=ConfigFile['Moco_lr'])
    lr_schedualer = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 80, 100], gamma=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Construct a training module:
    trainer = MoCoTrainer(Qencoder, Kencoder, Queue, loss_fn=loss_fn, optimizer=optimizer, scheduler=lr_schedualer, tau=ConfigFile['tau'],
                          cMomentum=ConfigFile['update_momentum'], flg=True, device=ConfigFile['device'])

    # Train the contrastive encoders scheme:
    trainer.train(dl_train=dl_train, dl_val=dl_val, epochs=ConfigFile['num_epochs'])

    '''Phase B - train Downstream task'''

def TrainDown_stream_task(ConfigFile):

    # Construct dataloaders:
    dl_train, dl_val, _ = DataModule(batch_size=ConfigFile['batch_size'], ks=ConfigFile['Gauss_kernel'],
                                              imagenet_stats=ConfigFile['imagenet_stats'])

    '''Initializing the Keys Encoder and SG with matching weights'''
    # Construct base encoder models:
    Qencoder = MoCo(backbone=models.resnet50(num_classes=ConfigFile['output_size']),
                    output_dim_list=ConfigFile['output_dim_list']).to(device=ConfigFile['device'])

    Qencoder.load_state_dict(torch.load(f'{os.getcwd()}/MoCoWeights/Model_Weights_epoch_40.pt'))

    # for p in Qencoder.parameters():
    #     p.requires_grad = False


    Classifier_model= DownStreamTaskModel(encoder=Qencoder, InputDim=128, OutputDim_List=[64, 32, 10]).to(device = ConfigFile['device'])
    # params= [{'params': model.fc.parameters(),'lr':ConfigFile['lr']}]
    # params= Classifier_model.fc.parameters()

    optimizer = torch.optim.Adam(params=Classifier_model.parameters(), lr = ConfigFile['Classifier_lr'])

    lr_schedualer = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80, 120], gamma=0.2)
    loss_fn = torch.nn.CrossEntropyLoss()

    trainer= DST_Trainer(model= Classifier_model,loss_fn=loss_fn,optimizer=optimizer,scheduler=lr_schedualer,device=ConfigFile['device'])
    trainer.train(dl_train=dl_train, dl_val=dl_val, epochs=ConfigFile['num_epochs'])


if __name__ == '__main__':

    # HyperParameters
    ConfigFile= dict(tau = 1e-5,
                     update_momentum = 0.999,
                     Moco_lr = 1e-5,
                     Classifier_lr = 1e-3,
                     queue_size = 9440,
                     batch_size = 32,
                     num_epochs = 150,
                     output_size = 128,
                     input_image_size = 224,
                     output_dim_list = [512, 128],
                     Gauss_kernel = (int(0.1 * 224) // 2) * 2 + 1,
                     imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
                     device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                     )

    TrainMocoContrastive(ConfigFile)
    # TrainDown_stream_task(ConfigFile)