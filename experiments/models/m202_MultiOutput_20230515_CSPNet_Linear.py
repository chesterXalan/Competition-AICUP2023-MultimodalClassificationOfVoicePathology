import torch
from torch import nn


def calc_padding(kernel_size, padding):
    if padding == 'same':
        return kernel_size//2
    elif padding == 'valid':
        return 0

def conv1d(Ci, Co, kernel_size, stride, padding):
    module = nn.Conv1d(Ci, Co,
                       kernel_size=kernel_size,
                       stride=stride,
                       padding=calc_padding(kernel_size, padding))
    nn.init.kaiming_normal_(module.weight) # He normal
    return module

def conv1d_bn_relu(Ci, Co, kernel_size, stride, padding='same'):
    module = nn.Sequential(
        conv1d(Ci, Co, kernel_size, stride, padding),
        nn.BatchNorm1d(Co),
        nn.ReLU(inplace=True)
    )
    return module

class SENetBlock(nn.Module):
    def __init__(self, Ci):
        super().__init__()
        Cm = Ci//4
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(),
            nn.Linear(Ci, Cm),
            nn.ReLU(inplace=True),
            nn.Linear(Cm, Ci),
            nn.Sigmoid(),
            nn.Unflatten(1, (Ci, 1)) # reshape Ci to Ci x 1
        )
        
    def forward(self, x):
        y = self.layers(x)
        y = torch.mul(x, y)
        return y

class ResNetBlock(nn.Module):
    def __init__(self, Ci):
        super().__init__()
        self.layers = nn.Sequential(
            conv1d_bn_relu(Ci, Ci, 3, 1),
            conv1d_bn_relu(Ci, Ci, 3, 1),
            SENetBlock(Ci)
        )
        
    def forward(self, x):
        y = self.layers(x)
        y = torch.add(x, y)
        return y

class CSPNetBlock(nn.Module):
    def __init__(self, Ci, Co):
        super().__init__()
        Cn = Ci//2
        self.layers1 = conv1d_bn_relu(Ci, Cn, 1, 1)
        self.layers2 = nn.Sequential(
            conv1d_bn_relu(Ci, Cn, 1, 1),
            ResNetBlock(Cn)
        )
        self.layers3 = conv1d_bn_relu(Ci, Co, 2, 2)
        
    def forward(self, x):
        y1 = self.layers1(x)
        y2 = self.layers2(x)
        y = torch.cat((y1, y2), 1)
        y = self.layers3(y)
        return y

class SignalClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            conv1d_bn_relu(1, 32, 11, 6, 'valid'),
            conv1d_bn_relu(32, 32, 3, 2, 'valid'),
            conv1d_bn_relu(32, 32, 3, 2, 'valid')
        )
        self.body = nn.Sequential(
            CSPNetBlock(32, 32),
            CSPNetBlock(32, 64),
            CSPNetBlock(64, 64)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 5)
        )
        
    def forward(self, x):
        y = self.stem(x)
        y = self.body(y)
        y1 = self.head(y)
        y2 = self.classifier(y1)
        return y1, y2


def linear_relu_dropout(in_features, out_features, p=0.25):
    module = nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.ReLU(inplace=True),
        nn.Dropout1d(p)
    )
    return module

class RecordClassifier(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.features = nn.Sequential(
            linear_relu_dropout(in_features, 128),
            linear_relu_dropout(128, 128),
            linear_relu_dropout(128, 64)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 5)
        )
        
    def forward(self, x):
        y1 = self.features(x)
        y2 = self.classifier(y1)
        return y1, y2


class Model(nn.Module):
    def __init__(self, rec_input_size):
        super().__init__()
        self.signal_classifier = SignalClassifier()
        self.record_classifier = RecordClassifier(rec_input_size)
        self.output_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 5)
        )
    
    def forward(self, x1, x2):
        y11, y12 = self.signal_classifier(x1)
        y21, y22 = self.record_classifier(x2)
        y = torch.cat((y11, y21), 1)
        y = self.output_classifier(y)
        return y, y12, y22
