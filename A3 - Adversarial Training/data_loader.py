import torch
import torchvision


def get_CIFAR10(train=True):
    if train :
        transform = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                                    torchvision.transforms.RandomCrop(32,padding=4),
                                                    torchvision.transforms.ToTensor()])
    else :
        transform = torchvision.transforms.ToTensor()
                                                
    return torchvision.datasets.CIFAR10(root='./datas/CIFAR10/',
                                        train=train,
                                        transform=transform,
                                        download=True)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, datas, settings):
        self.inputs = []
        self.outputs = []
        self.size = 0

        for x, y in datas:
            self.inputs.append(x)
            self.outputs.append(y)
            self.size += 1

        self.inputs = torch.stack(self.inputs).to(**settings)
        self.outputs = torch.Tensor(self.outputs).to(device=settings['device'], dtype=torch.int64)

    def __len__(self): return self.size
    def __getitem__(self, idx): return(self.inputs[idx], self.outputs[idx])


class CIFAR10:
    def __init__(self, train, settings):
        self.datas = MyDataset(get_CIFAR10(train), settings)

    def make(self, batch_size):
        return torch.utils.data.DataLoader(
            self.datas, batch_size=batch_size, shuffle=True)
