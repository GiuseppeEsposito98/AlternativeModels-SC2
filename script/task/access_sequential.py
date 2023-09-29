import torch
from torchvision import models

from torch import nn

layers = [nn.Conv2d(3, 32, (1,1)), nn.Conv2d(32, 64, (1,1)), nn.Conv2d(32, 64, (1,1)), nn.Conv2d(32, 64, (1,1))]
model = nn.Sequential(*layers)

# print(models.detection.__dict__)


from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights

ssd = ssd300_vgg16(pretrained=False, pretrained_backbone=False)


def rec(model):
    for idx, module in ssd.named_children():
        if idx == 'features':
            print(module.state_dict())
        else: 
            rec(module)

# print(ssd.backbone.features.state_dict())

# def validate_state_dicts(k_1, k_2):
    # if len(model_state_dict_1) != len(model_state_dict_2):
    #     print(
    #         f"Length mismatch: {len(model_state_dict_1)}, {len(model_state_dict_2)}"
    #     )
    #     return False

    # Replicate modules have "module" attached to their keys, so strip these off when comparing to local model.
    # if next(iter(model_state_dict_1.keys())).startswith("module"):
    #     model_state_dict_1 = {
    #         k[len("module") + 1 :]: v for k, v in model_state_dict_1.items()
    #     }

    # if next(iter(model_state_dict_2.keys())).startswith("module"):
    #     model_state_dict_2 = {
    #         k[len("module") + 1 :]: v for k, v in model_state_dict_2.items()
    #     }

    # for ((k_1, v_1), (k_2, v_2)) in zip(
    #     model_state_dict_1.items(), model_state_dict_2.items()
    # ):
    #     if k_1 != k_2:
    #         print(f"Key mismatch: {k_1} vs {k_2}")
    #         return False
    # # convert both to the same CUDA device
    # if str(k_1.device) != "cuda:0":
    #     k_1 = k_1.to("cuda:0" if torch.cuda.is_available() else "cpu")
    # if str(k_2.device) != "cuda:0":
    #     k_2 = k_2.to("cuda:0" if torch.cuda.is_available() else "cpu")

    # if not torch.allclose(k_1.to('cpu'), k_2.to('cpu')):
    #     print(f"Tensor mismatch: {k_1.to('cpu')} vs {k_2.to('cpu')}")
    #     return False
        

ref_model = torch.load('/home/g.esposito/sc2-benchmark/script/task/weights_ref.pt')
model = torch.load('/home/g.esposito/sc2-benchmark/script/task/weights.pt')

# print(f'model: {ref_model.keys()}')
# print(f'model: {model.keys()}')

# print(validate_state_dicts(model['backbone.layer1.0.bias'].to('cpu'),ref_model['backbone.features.0.bias'].to('cpu')))

print(torch.allclose(model['backbone.layer1.0.weight'].to('cpu'),ref_model['backbone.features.0.weight'].to('cpu')))