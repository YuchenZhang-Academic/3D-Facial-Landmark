import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class Net(nn.Module):
    
    def __init__(self, output_size=8):
        super(Net, self).__init__()
        self.output_size = output_size
        self.dropout = nn.Dropout(p=0.8)
        self.Conv_layers = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=3, kernel_size=4, stride=2),
            nn.MaxPool3d(kernel_size=2, stride=1),
            nn.BatchNorm3d(3),
            nn.Sigmoid()
            # 98, 98, 98
            )
        self.res = models.resnet18(pretrained=False)
        # self.res.load_state_dict(torch.load('./resnet18-5c106cde.pth'))
        self.Linear = nn.Sequential(
            nn.Linear(1000, 512),
            nn.Sigmoid(),
            nn.Linear(512, 256),
            nn.Sigmoid(),
            nn.Linear(256, output_size * 3),
            nn.Sigmoid()
            )

    def forward(self, input):
        x = self.Conv_layers(input)
        x = x.view(-1, 3, 98, 98*98)
        x = self.res(x)
        # x = self.dropout(x)
        x = self.Linear(x)
        self.pred = x.reshape(-1, self.output_size, 3)

        return self.pred
    
    def cal_loss(self, labels):
        n_batch = labels.size()[0]
        self.pred = self.pred.double()
        labels = (labels.double() + 1)
        crition = nn.MSELoss()
        loss = 0
        temp_min = 1000000
        temp_max = 0
        for k in range(n_batch):
            loss += torch.sqrt(crition(labels[k], self.pred[k]))
            for i in range(self.output_size):
                t = torch.sqrt(crition(labels[k][i], self.pred[k][i]))
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



    

if __name__ == '__main__':
    x = torch.zeros(1, 1, 201, 201, 201)
    net = Net(output_size=8)
    a = net(x)
    label = torch.zeros(1, 8, 3)
    print(a[0])
    print(net.cal_loss(label))
