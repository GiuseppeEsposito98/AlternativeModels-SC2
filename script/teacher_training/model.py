import torch
from torchvision import models

class BackboneUnpacker(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.original_model = model
        self.unpack(model.features)
        self.avgpool = model.avgpool
        self.classifier = model.classifier

    
    def unpack(self, original_model):
        modules = list()
        for blocks in original_model.features.children():
            modules.append(blocks)
        return modules
    
    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        return x

            
