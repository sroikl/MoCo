from torchvision.datasets.utils import download_url
import os
import tarfile
import hashlib
import torchvision
import torch
from torchvision.transforms import transforms
import random
from PIL import ImageFilter

def DataModule(batch_size,ks,imagenet_stats):

    # https://github.com/fastai/imagenette
    dataset_url = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz'
    dataset_filename = dataset_url.split('/')[-1]
    dataset_foldername = dataset_filename.split('.')[0]
    data_path = './data'
    dataset_filepath = os.path.join(data_path, dataset_filename)
    dataset_folderpath = os.path.join(data_path, dataset_foldername)

    os.makedirs(data_path, exist_ok=True)

    download = False
    if not os.path.exists(dataset_filepath):
        download = True
    else:
        md5_hash = hashlib.md5()

        file = open(dataset_filepath, "rb")

        content = file.read()

        md5_hash.update(content)

        digest = md5_hash.hexdigest()
        if digest != 'fe2fc210e6bb7c5664d602c3cd71e612':
            download = True
    if download:
        download_url(dataset_url, data_path)

    with tarfile.open(dataset_filepath, 'r:gz') as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=data_path)


    train_transform = TwoCropsTransform(transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**imagenet_stats)
        ]))

    dataset_train = torchvision.datasets.ImageFolder(os.path.join(dataset_folderpath, 'train'), train_transform)
    dataset_validation = torchvision.datasets.ImageFolder(os.path.join(dataset_folderpath, 'val'), train_transform)

    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        num_workers=4,
        drop_last=True,
        shuffle=True,
    )

    validation_dataloader = torch.utils.data.DataLoader(
        dataset_validation,
        batch_size=batch_size,
        num_workers=4,
        drop_last=True,
        shuffle=True,
    )
    return train_dataloader,validation_dataloader,transforms


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

    def __repr__(self):
        format_string = self.__class__.__name__ + '(\n\t'
        format_string += self.base_transform.__repr__().replace('\n', '\n\t')
        format_string += '\n)'
        return format_string

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x