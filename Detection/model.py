import torch
import torch.nn as nn
import torchvision.models as tvmodel
import math

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        '''
        TODO:BackBone
        '''
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
            nn.Linear(8192, 6 * 6),
            nn.Sigmoid()
        )

    def forward(self, input):
        x = self.Conv_layers(input)
        x = x.view(x.size()[0], -1)
        x = self.connection_layers(x)
        self.pred = x.reshape(-1, 6, 6)

        return self.pred
    
    def cal_loss(self, labels):
        n_batch = labels.size()[0]
        self.pred = self.pred.double()

        labels = (labels.double() + 1) / 2

        loss = 0
        lambda_1 = 0.5
        lambda_2 = 1 - lambda_1
        for k in range(n_batch):
            for i in range(6):
                loss += (lambda_1 * ((self.pred[k][i][0] - labels[k][i][0])**2 + (self.pred[k][i][1] - labels[k][i][1])**2 + (self.pred[k][i][2] - labels[k][i][2])**2) + lambda_2 * ((math.sqrt(self.pred[k][i][3]) - math.sqrt(labels[k][i][3]))**2 + (math.sqrt(self.pred[k][i][4]) - math.sqrt(labels[k][i][4]))**2 + (math.sqrt(self.pred[k][i][5]) - math.sqrt(labels[k][i][5]))**2))

        return loss / n_batch
        '''
        TODO 
        或者直接改成欧氏距离
        '''
    

if __name__ == '__main__':
    x = torch.zeros(4, 1, 201, 201, 201)
    net = Net()
    a = net(x)
    label = torch.zeros(1, 6, 6)
    print(a[0])
    print(net.cal_loss(label))
