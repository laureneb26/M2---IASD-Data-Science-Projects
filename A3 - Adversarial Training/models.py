
import torch
import torch.nn.functional as F

class GaussianNoise(torch.nn.Module) :
    def __init__(self,sigma,device='cuda'):
        super().__init__()
        self.sigma = sigma
        print ( device )
        self.noise = torch.tensor(0.).to(device)
    
    def forward(self,x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach()
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x

class ConvNet(torch.nn.Module):
    def __init__(self,dropout,sigma,settings):
        super(ConvNet, self).__init__()
        self.dropout = dropout
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=2)
        self.noise1 = GaussianNoise(sigma)
        self.conv2 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=2)
        self.conv3 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=2)

        self.lin1 = torch.nn.Linear(512 * 2 * 2, 1024)
        self.lin2 = torch.nn.Linear(1024, 10)
        
        self.latent = False

    def forward(self, x):
        y = self.noise1(self.conv1(x))
        y = F.max_pool2d(F.relu(y), 2)
        if self.dropout : y = F.dropout ( y, p = self.dropout, training = self.training )
        y = F.max_pool2d(F.relu(self.conv2(y)), 2)
        if self.dropout : y = F.dropout ( y, p = self.dropout, training = self.training )
        y = F.max_pool2d(F.relu(self.conv3(y)), 2)

        y = F.avg_pool2d(y, 2)

        y = y.view(-1, 512 * 2 * 2)
        y = F.relu(self.lin1(y))
        z = self.lin2(y)
        if self.latent : return (y, z)
        return z
    
    def set_latent(self,latent=True):
        self.latent = latent