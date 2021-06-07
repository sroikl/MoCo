from DataModule import DataModule
from Model import MoCo #, DownStreamTaskModel
import torch
from Trainer import MoCoTrainer, DictionaryQueue
import torchvision.models as models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# print(f'Working with {device}, number of GPUs: {torch.cuda.device_count()}')

def model_pipeline():
    '''Phase A - train Momentum Encoder'''

    # HyperParameters
    tau = 0.05  # Contrastive loss temperament
    update_momentum = 0.9
    lr = 1e-4
    queue_size = 4096
    batch_size = 32
    num_epochs = 200
    output_size = 256
    input_image_size = 224
    output_dim_list= [512,128
                      ]
    # Data augmentation parameters:
    Gauss_kernel = (int(0.1 * input_image_size) // 2) * 2 + 1  # should be odd
    imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

    # Define model backbone for the base encoders:

    # Construct and Initialize dictionary queue:
    Queue = DictionaryQueue(output_size, queue_size, device)
    # Construct dataloaders:
    dl_train, dl_val, transforms = DataModule(batch_size=batch_size, ks=Gauss_kernel, imagenet_stats=imagenet_stats)

    '''Initializing the Keys Encoder and SG with matching weights'''
    # Construct base encoder models:
    Qencoder = MoCo(backbone=models.resnet50(num_classes=output_size),output_dim_list=output_dim_list).to(device=device)
    Kencoder = MoCo(backbone=models.resnet50(num_classes=output_size),output_dim_list=output_dim_list).to(device=device)
    # Match the weights:
    for param_q, param_k in zip(Qencoder.parameters(), Kencoder.parameters()):
        param_k.data.copy_(param_q.data)

    # Define the optimization process:
    # optimizer = torch.optim.SGD(params=Qencoder.parameters(), lr=lr, weight_decay=1e-4, momentum=0.9)
    optimizer = torch.optim.Adam(params=Qencoder.parameters(), lr=lr)
    lr_schedualer = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120, 160], gamma=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Construct a training module:
    trainer = MoCoTrainer(Qencoder, Kencoder, Queue, loss_fn=loss_fn, optimizer=optimizer, scheduler=lr_schedualer, tau=tau,
                          cMomentum=update_momentum, flg=True, device=device)

    # Train the contrastive encoders scheme:
    trainer.train(dl_train=dl_train, dl_val=dl_val, epochs=num_epochs)

    '''Phase B - train Downstream task'''


if __name__ == '__main__':
    model_pipeline()
