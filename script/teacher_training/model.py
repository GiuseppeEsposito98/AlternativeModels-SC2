import torch
from torchvision import models

class BackboneUnpacker(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.original_model = model
        self.unpack(model.features)
        self.avgpool = model.avgpool
        self.classifier = model.classifier

    
    def unpack(self):
        for name, module in self.original_model.named_children():
            if name == 'features':
                for idx, blocks in enumerate(module.children()):
                    exec(f'self.layer{idx} = blocks')
    
    def forward():
        pass

            
