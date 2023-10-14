from torchvision.models.mobilenetv3 import MobileNetV3, _mobilenet_v3_conf




def main():
    # model instantiation
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch='mobilenet_v3_small')
    model = MobileNetV3(inverted_residual_setting=inverted_residual_setting, last_channel=last_channel)

    # dataloaders
    

    # train

    # eval

if __name__ == '__main__':
    main()