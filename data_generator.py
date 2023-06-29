import torch
from torchvision.datasets import MNIST
from torchvision.transforms import (
    Resize,
    Normalize,
    Compose,
    RandomHorizontalFlip,
    ToTensor
)
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def get_random_colored_images(images, seed = 0x000000):
    np.random.seed(seed)
    
    images = 0.5*(images + 1)
    size = images.shape[0]
    colored_images = []
    hues = 360*np.random.rand(size)
    
    for V, H in zip(images, hues):
        V_min = 0
        
        a = (V - V_min)*(H%60)/60
        V_inc = a
        V_dec = V - a
        
        colored_image = torch.zeros((3, V.shape[1], V.shape[2]))
        H_i = round(H/60) % 6
        
        if H_i == 0:
            colored_image[0] = V
            colored_image[1] = V_inc
            colored_image[2] = V_min
        elif H_i == 1:
            colored_image[0] = V_dec
            colored_image[1] = V
            colored_image[2] = V_min
        elif H_i == 2:
            colored_image[0] = V_min
            colored_image[1] = V
            colored_image[2] = V_inc
        elif H_i == 3:
            colored_image[0] = V_min
            colored_image[1] = V_dec
            colored_image[2] = V
        elif H_i == 4:
            colored_image[0] = V_inc
            colored_image[1] = V_min
            colored_image[2] = V
        elif H_i == 5:
            colored_image[0] = V
            colored_image[1] = V_min
            colored_image[2] = V_dec
        
        colored_images.append(colored_image)
        
    colored_images = torch.stack(colored_images, dim = 0)
    colored_images = 2*colored_images - 1
    
    return colored_images


class DataGenerator:
    def __init__(self, config):
        self.config = config
        
        self.train_mnist_transforms = Compose(
            [
                # Resize((config.data.image_size, config.data.image_size)),
                Resize((32, 32)),
                ToTensor(),
                Normalize(mean=config.data.norm_mean, std=config.data.norm_std),
                # to [-1; 1]
            ]
        )
        
        
        self.valid_mnist_transforms = Compose(
            [
                # Resize((config.data.image_size, config.data.image_size)),
                Resize((32, 32)),
                ToTensor(),
                Normalize(mean=config.data.norm_mean, std=config.data.norm_std),
                # to [-1; 1]
            ]
        )
        
        dataset_name = config.data.dataset.split("_")[0]
        is_colored = dataset_name[-7:] == "colored"
        classes = [int(number) for number in config.data.dataset.split("_")[1:]]
        if not classes:
            classes = [i for i in range(10)]
        
        train_set =  MNIST(config.data.path, train=True, transform=self.train_mnist_transforms, download=True)
        test_set =  MNIST(config.data.path, train=False, transform=self.valid_mnist_transforms, download=True)
        
        train_test = []
        for dataset in [train_set, test_set]:
            data = []
            labels = []
            for k in range(len(classes)):
                data.append(torch.stack(
                    [dataset[i][0] for i in range(len(dataset.targets)) if dataset.targets[i] == classes[k]],
                    dim=0
                ))
                labels += [k]*data[-1].shape[0]
            data = torch.cat(data, dim=0)
            data = data.reshape(-1, 1, 32, 32)
            labels = torch.tensor(labels)
            
            if is_colored:
                data = get_random_colored_images(data)
            
            train_test.append(TensorDataset(data, labels))
            
        train_set, test_set = train_test        
        
        
        
        self.train_loader = DataLoader(
            train_set,
            batch_size=config.training.batch_size,
            shuffle=True,
            drop_last=True
        )
        
        
        self.valid_loader = DataLoader(
            test_set,
            batch_size= config.training.batch_size,
            shuffle=False,
            drop_last=False
        )

    def sample_train(self):
        while True:
            for batch in self.train_loader:
                yield batch

                
                
                
"""
class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.mnist_transforms = Compose(
            [
                Resize((config.data.image_size, config.data.image_size)),
                ToTensor(),
                Normalize(mean=config.data.norm_mean, std=config.data.norm_std),
                # to [-1; 1]
            ]
        )
        self.train_loader = DataLoader(
            MNIST(root='/home/mounted/data/train_MNIST', download=True, train=True, transform=self.mnist_transforms),
            batch_size=config.training.batch_size,
            shuffle=True,
            drop_last=True
        )
        self.valid_loader = DataLoader(
            MNIST(root='/home/mounted/data/train_MNIST', download=True, train=False, transform=self.mnist_transforms),
            batch_size=5*config.training.batch_size,
            shuffle=False,
            drop_last=False
        )

    def sample_train(self):
        while True:
            for batch in self.train_loader:
                yield batch
"""