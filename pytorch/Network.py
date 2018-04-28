import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Tracknet(nn.Module):
    '''
        actor network
        input: states (N*112*112*3)
        output: action (N*11)
    '''
    def __init__(self):
        super().__init__()
        pretrained_model = torchvision.models.resnet18()
        self.features = nn.Sequential(
            *list(pretrained_model.features.children())[:-3]
        )
        self.fc = nn.Sequential(
            nn.Linear(3*3*512, 512),
            nn.Tanh(),
            nn.Linear(512, 11)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        x = F.softmax(x)
        return x
