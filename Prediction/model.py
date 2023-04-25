import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class Net(nn.Module):
    
    def __init__(self, output_size=8):
        super(Net, self).__init__()
        '''
        TODO:BackBone
        '''
        self.output_size = output_size
        # self.res = models.resnet18(pretrained=False)
        # self.res.load_state_dict(torch.load('./resnet18-5c106cde.pth'))
        self.Conv_layers = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=4, stride=2),
            nn.MaxPool3d(kernel_size=2, stride=1),
            nn.BatchNorm3d(num_features=1),
            nn.Sigmoid(),
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=4, stride=2),
            nn.MaxPool3d(kernel_size=2, stride=1),
            nn.BatchNorm3d(num_features=1),
            nn.Sigmoid(),
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=4, stride=2),
            nn.MaxPool3d(kernel_size=2, stride=1),
            nn.BatchNorm3d(num_features=1),
            nn.Sigmoid(),
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=4, stride=1),
            nn.MaxPool3d(kernel_size=2, stride=1),
            nn.BatchNorm3d(num_features=1),
            nn.Sigmoid()
            )
        self.connection_layers = nn.Sequential(
            nn.Linear(17 * 17 * 17 * 1 * 1, 8192),
            nn.Sigmoid(),
            nn.Linear(8192, 4096),
            nn.Sigmoid(),
            nn.Linear(4096, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, self.output_size * 3),
            nn.Sigmoid()
        )

    def forward(self, input):
        x = self.Conv_layers(input)
        x = x.view(x.size()[0], -1)
        # x = self.res(x)
        x = self.connection_layers(x)
        self.pred = x.reshape(-1, self.output_size, 3)

        return self.pred
    
    def cal_loss(self, labels):
        n_batch = labels.size()[0]
        self.pred = self.pred.double()

        # 将label的范围由[-1, 1]转变为[0, 1]
        labels = (labels.double() + 1) / 2
        crition = nn.MSELoss()
        loss = 0
        temp_min = 1000000
        temp_max = 0
        for k in range(n_batch):
            loss += torch.sqrt(crition(labels[k], self.pred[k]))
            for i in range(self.output_size):
                t = torch.sqrt((labels[k][i][0] - self.pred[k][i][0]) ** 2 + (labels[k][i][1] - self.pred[k][i][1]) ** 2 + (labels[k][i][2] - self.pred[k][i][2]) ** 2)
                if t.item() > temp_max:
                    temp_max = t.item()
                if t.item() < temp_min:
                    temp_min = t.item()

        return loss / n_batch, temp_max, temp_min
    
    def cal_inf_loss(self, labels):
        n_batch = labels.size()[0]
        self.pred = self.pred.double()

        # 将label的范围由[-1, 1]转变为[0, 1]
        labels = (labels.double() + 1) / 2
        crition = nn.MSELoss()
        loss = []
        for k in range(n_batch):
            max_loss = torch.Tensor([0])
            for i in range(self.output_size):
                temp = torch.sqrt(crition(labels[k][i], self.pred[k][i]))
                if  temp.item() > max_loss.item():
                    max_loss = temp
            loss.append(max_loss)
        return torch.stack(loss, dim=0).mean(dim=0), 0, 0

    def cal_test_loss(self, labels):
        n_batch = labels.size()[0]
        self.pred = self.pred.double()

        # 将label的范围由[-1, 1]转变为[0, 1]
        labels = (labels.double() + 1) / 2
        crition = nn.MSELoss()
        loss = []
        for k in range(n_batch):
            temp = []
            for i in range(self.output_size):
                temp.append(torch.sqrt(crition(labels[k][i], self.pred[k][i])).item())
            loss.append(temp)
        return loss



    

if __name__ == '__main__':
    x = torch.zeros(1, 1, 201, 201, 201)
    net = Net(output_size=8)
    a = net(x)
    label = torch.zeros(1, 8, 3)
    print(a[0])
    print(net.cal_inf_loss(label))
