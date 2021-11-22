import torch.nn as nn
import torch.nn.functional as functional
import torch.nn.init as init

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fea_conv1 = nn.Conv2d(1, 32, 3, 1, 1, bias=True)
        self.resblock1 = make_layer(ResidualBlock, 32, 3)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.dropout1 = nn.Dropout2d(0.5)

        self.fea_conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=True)
        self.resblock2 = make_layer(ResidualBlock, 64, 3)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.dropout2 = nn.Dropout2d(0.5)

        self.fea_conv3 = nn.Conv2d(64, 32, 3, 1, 1, bias=True)
        self.resblock3 = make_layer(ResidualBlock, 32, 3)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)
        self.dropout3 = nn.Dropout2d(0.5)

        self.fc1 = nn.Linear(in_features=32*7*7,out_features=1568)
        self.fc2 = nn.Linear(in_features=1568,out_features=784)
        self.out = nn.Linear(in_features=784,out_features=36)
        
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):   

        x = self.relu(self.fea_conv1(x)) # [1, 32, 56, 56]
        print(x.shape)
        x = self.resblock1(x) # [1, 32, 56, 56]
        print(x.shape)
        x = self.maxpool1(x) # [1, 32, 28, 28]
        print(x.shape)
        x = self.dropout1(x) # [1, 32, 28, 28]
        print(x.shape)
        print("=====================")

        x = self.relu(self.fea_conv2(x)) # [1, 64, 28, 28]
        print(x.shape)
        x = self.resblock2(x) # [1, 64, 28, 28]
        print(x.shape)
        x = self.maxpool2(x) # [1, 64, 14, 14]
        print(x.shape)
        x = self.dropout2(x) # [1, 64, 14, 14]
        print(x.shape)
        print("=====================")

        x = self.relu(self.fea_conv3(x)) # [1, 32, 14, 14]
        print(x.shape)
        x = self.resblock3(x) # [1, 32, 14, 14]
        print(x.shape)
        x = self.maxpool3(x) # [1, 32, 7, 7]
        print(x.shape)
        x = self.dropout3(x) # [1, 32, 7, 7]
        print(x.shape)
        print("=====================")

        x = x.reshape(-1, 32*7*7) 

        x = self.relu(self.fc1(x))
        print(x.shape) # [1, 1568]
        x = self.relu(self.fc2(x))
        print(x.shape) # [1, 784]
        out = self.out(x)
        print(out.shape) # [1, 36]
        print(out)
        print("=====================")

        return out



class ResidualBlock(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return x + out


def make_layer(block, nf, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block(nf))
    return nn.Sequential(*layers)


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()