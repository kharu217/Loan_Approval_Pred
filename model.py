import torch
import torch.nn as nn

class loan_classify(nn.Module) :
    def __init__(self):
        super(loan_classify, self).__init__()
        self.layer1 = nn.Linear(9, 18)
        self.layer2 = nn.Linear(18, 18)
        self.layer3 = nn.Linear(18, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x) :
        x = self.sigmoid(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x
