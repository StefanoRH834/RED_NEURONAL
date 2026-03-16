import torchvision.transforms as T
from torchvision.datasets import CIFAR10
transforms = T.Compose([T.ToTensor(),
                        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
train_data = CIFAR10(root='./train/', train=True, download=True, transform=transforms)
test_data = CIFAR10(root='./test/', train=False, download=True, transform=transforms)