# SSD(
#   (backbone): SSDFeatureExtractorVGG(
#     (features): Sequential(
#       (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (1): ReLU(inplace=True)
#       (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (3): ReLU(inplace=True)
#       (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#       (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (6): ReLU(inplace=True)
#       (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (8): ReLU(inplace=True)
#       (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#       (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (11): ReLU(inplace=True)
#       (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (13): ReLU(inplace=True)
#       (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (15): ReLU(inplace=True)
#       (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
#       (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (18): ReLU(inplace=True)
#       (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (20): ReLU(inplace=True)
#       (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (22): ReLU(inplace=True)
#     )
#     (extra): ModuleList(
#       (0): Sequential(
#         (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#         (1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (2): ReLU(inplace=True)
#         (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (4): ReLU(inplace=True)
#         (5): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (6): ReLU(inplace=True)
#         (7): Sequential(
#           (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
#           (1): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(6, 6), dilation=(6, 6))
#           (2): ReLU(inplace=True)
#           (3): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
#           (4): ReLU(inplace=True)
#         )
#       )
#       (1): Sequential(
#         (0): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
#         (1): ReLU(inplace=True)
#         (2): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#         (3): ReLU(inplace=True)
#       )
#       (2): Sequential(
#         (0): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
#         (1): ReLU(inplace=True)
#         (2): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#         (3): ReLU(inplace=True)
#       )
#       (3): Sequential(
#         (0): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
#         (1): ReLU(inplace=True)
#         (2): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
#         (3): ReLU(inplace=True)
#       )
#       (4): Sequential(
#         (0): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
#         (1): ReLU(inplace=True)
#         (2): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
#         (3): ReLU(inplace=True)
#       )
#     )
#   )
#   (anchor_generator): DefaultBoxGenerator(aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]], clip=True, scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05], steps=[8, 16, 32, 64, 100, 300])
#   (head): SSDHead(
#     (classification_head): SSDClassificationHead(
#       (module_list): ModuleList(
#         (0): Conv2d(512, 364, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (1): Conv2d(1024, 546, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (2): Conv2d(512, 546, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (3): Conv2d(256, 546, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (4): Conv2d(256, 364, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (5): Conv2d(256, 364, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       )
#     )
#     (regression_head): SSDRegressionHead(
#       (module_list): ModuleList(
#         (0): Conv2d(512, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (1): Conv2d(1024, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (2): Conv2d(512, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (3): Conv2d(256, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (4): Conv2d(256, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (5): Conv2d(256, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       )
#     )
#   )
#   (transform): GeneralizedRCNNTransform(
#       Normalize(mean=[0.48235, 0.45882, 0.40784], std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])
#       Resize(min_size=(300,), max_size=300, mode='bilinear')
#   )
# )




# FasterRCNN(
#   (transform): GeneralizedRCNNTransform(
#       Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#       Resize(min_size=(800,), max_size=1333, mode='bilinear')
#   )
#   (backbone): BackboneWithFPN(
#     (body): IntermediateLayerGetter(
#       (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#       (bn1): FrozenBatchNorm2d(64, eps=0.0)
#       (relu): ReLU(inplace=True)
#       (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#       (layer1): Sequential(
#         (0): Bottleneck(
#           (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn1): FrozenBatchNorm2d(64, eps=0.0)
#           (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#           (bn2): FrozenBatchNorm2d(64, eps=0.0)
#           (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn3): FrozenBatchNorm2d(256, eps=0.0)
#           (relu): ReLU(inplace=True)
#           (downsample): Sequential(
#             (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#             (1): FrozenBatchNorm2d(256, eps=0.0)
#           )
#         )
#         (1): Bottleneck(
#           (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn1): FrozenBatchNorm2d(64, eps=0.0)
#           (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#           (bn2): FrozenBatchNorm2d(64, eps=0.0)
#           (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn3): FrozenBatchNorm2d(256, eps=0.0)
#           (relu): ReLU(inplace=True)
#         )
#         (2): Bottleneck(
#           (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn1): FrozenBatchNorm2d(64, eps=0.0)
#           (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#           (bn2): FrozenBatchNorm2d(64, eps=0.0)
#           (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn3): FrozenBatchNorm2d(256, eps=0.0)
#           (relu): ReLU(inplace=True)
#         )
#       )
#       (layer2): Sequential(
#         (0): Bottleneck(
#           (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn1): FrozenBatchNorm2d(128, eps=0.0)
#           (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#           (bn2): FrozenBatchNorm2d(128, eps=0.0)
#           (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn3): FrozenBatchNorm2d(512, eps=0.0)
#           (relu): ReLU(inplace=True)
#           (downsample): Sequential(
#             (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
#             (1): FrozenBatchNorm2d(512, eps=0.0)
#           )
#         )
#         (1): Bottleneck(
#           (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn1): FrozenBatchNorm2d(128, eps=0.0)
#           (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#           (bn2): FrozenBatchNorm2d(128, eps=0.0)
#           (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn3): FrozenBatchNorm2d(512, eps=0.0)
#           (relu): ReLU(inplace=True)
#         )
#         (2): Bottleneck(
#           (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn1): FrozenBatchNorm2d(128, eps=0.0)
#           (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#           (bn2): FrozenBatchNorm2d(128, eps=0.0)
#           (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn3): FrozenBatchNorm2d(512, eps=0.0)
#           (relu): ReLU(inplace=True)
#         )
#         (3): Bottleneck(
#           (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn1): FrozenBatchNorm2d(128, eps=0.0)
#           (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#           (bn2): FrozenBatchNorm2d(128, eps=0.0)
#           (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn3): FrozenBatchNorm2d(512, eps=0.0)
#           (relu): ReLU(inplace=True)
#         )
#       )
#       (layer3): Sequential(
#         (0): Bottleneck(
#           (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn1): FrozenBatchNorm2d(256, eps=0.0)
#           (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#           (bn2): FrozenBatchNorm2d(256, eps=0.0)
#           (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn3): FrozenBatchNorm2d(1024, eps=0.0)
#           (relu): ReLU(inplace=True)
#           (downsample): Sequential(
#             (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
#             (1): FrozenBatchNorm2d(1024, eps=0.0)
#           )
#         )
#         (1): Bottleneck(
#           (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn1): FrozenBatchNorm2d(256, eps=0.0)
#           (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#           (bn2): FrozenBatchNorm2d(256, eps=0.0)
#           (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn3): FrozenBatchNorm2d(1024, eps=0.0)
#           (relu): ReLU(inplace=True)
#         )
#         (2): Bottleneck(
#           (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn1): FrozenBatchNorm2d(256, eps=0.0)
#           (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#           (bn2): FrozenBatchNorm2d(256, eps=0.0)
#           (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn3): FrozenBatchNorm2d(1024, eps=0.0)
#           (relu): ReLU(inplace=True)
#         )
#         (3): Bottleneck(
#           (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn1): FrozenBatchNorm2d(256, eps=0.0)
#           (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#           (bn2): FrozenBatchNorm2d(256, eps=0.0)
#           (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn3): FrozenBatchNorm2d(1024, eps=0.0)
#           (relu): ReLU(inplace=True)
#         )
#         (4): Bottleneck(
#           (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn1): FrozenBatchNorm2d(256, eps=0.0)
#           (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#           (bn2): FrozenBatchNorm2d(256, eps=0.0)
#           (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn3): FrozenBatchNorm2d(1024, eps=0.0)
#           (relu): ReLU(inplace=True)
#         )
#         (5): Bottleneck(
#           (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn1): FrozenBatchNorm2d(256, eps=0.0)
#           (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#           (bn2): FrozenBatchNorm2d(256, eps=0.0)
#           (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn3): FrozenBatchNorm2d(1024, eps=0.0)
#           (relu): ReLU(inplace=True)
#         )
#       )
#       (layer4): Sequential(
#         (0): Bottleneck(
#           (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn1): FrozenBatchNorm2d(512, eps=0.0)
#           (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#           (bn2): FrozenBatchNorm2d(512, eps=0.0)
#           (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn3): FrozenBatchNorm2d(2048, eps=0.0)
#           (relu): ReLU(inplace=True)
#           (downsample): Sequential(
#             (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
#             (1): FrozenBatchNorm2d(2048, eps=0.0)
#           )
#         )
#         (1): Bottleneck(
#           (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn1): FrozenBatchNorm2d(512, eps=0.0)
#           (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#           (bn2): FrozenBatchNorm2d(512, eps=0.0)
#           (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn3): FrozenBatchNorm2d(2048, eps=0.0)
#           (relu): ReLU(inplace=True)
#         )
#         (2): Bottleneck(
#           (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn1): FrozenBatchNorm2d(512, eps=0.0)
#           (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#           (bn2): FrozenBatchNorm2d(512, eps=0.0)
#           (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (bn3): FrozenBatchNorm2d(2048, eps=0.0)
#           (relu): ReLU(inplace=True)
#         )
#       )
#     )
#     (fpn): FeaturePyramidNetwork(
#       (inner_blocks): ModuleList(
#         (0): Conv2dNormActivation(
#           (0): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (1): Conv2dNormActivation(
#           (0): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (2): Conv2dNormActivation(
#           (0): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (3): Conv2dNormActivation(
#           (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (layer_blocks): ModuleList(
#         (0): Conv2dNormActivation(
#           (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         )
#         (1): Conv2dNormActivation(
#           (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         )
#         (2): Conv2dNormActivation(
#           (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         )
#         (3): Conv2dNormActivation(
#           (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         )
#       )
#       (extra_blocks): LastLevelMaxPool()
#     )
#   )
#   (rpn): RegionProposalNetwork(
#     (anchor_generator): AnchorGenerator()
#     (head): RPNHead(
#       (conv): Sequential(
#         (0): Conv2dNormActivation(
#           (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (1): ReLU(inplace=True)
#         )
#       )
#       (cls_logits): Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
#       (bbox_pred): Conv2d(256, 12, kernel_size=(1, 1), stride=(1, 1))
#     )
#   )
#   (roi_heads): RoIHeads(
#     (box_roi_pool): MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=(7, 7), sampling_ratio=2)
#     (box_head): TwoMLPHead(
#       (fc6): Linear(in_features=12544, out_features=1024, bias=True)
#       (fc7): Linear(in_features=1024, out_features=1024, bias=True)
#     )
#     (box_predictor): FastRCNNPredictor(
#       (cls_score): Linear(in_features=1024, out_features=91, bias=True)
#       (bbox_pred): Linear(in_features=1024, out_features=364, bias=True)
#     )
#   )
# )


# outputs: [{'boxes': tensor([[128.8358,  88.6164, 517.2512, 534.4267],
#         [507.2096, 316.0583, 567.9502, 385.7333],
#         [ 76.4161, 375.3253,  97.1108, 427.8564],
#         [189.3943, 307.2538, 250.3843, 368.8841],
#         [113.7504, 383.5127, 128.5445, 430.6164],
#         [208.8653,  21.3145, 226.0974,  43.1617],
#         [ 99.7850, 397.1490, 115.4947, 429.8177],
#         [176.9726, 166.8015, 201.2966, 192.9639],
#         [571.1475, 322.2679, 611.4950, 403.0961],
#         [ 91.0663, 390.8993, 103.2760, 430.0912],
#         [397.0776, 178.7671, 411.5399, 198.9274],
#         [569.8697, 337.9153, 610.4628, 402.6506],
#         [ 61.6254, 354.4955,  94.4995, 368.5665],
#         [177.7504, 177.7354, 198.1568, 192.8504],
#         [301.4918, 319.2320, 348.7804, 366.3032],
#         [115.3095, 377.2700, 126.6362, 412.8477],
#         [569.5651, 323.8925, 610.7169, 403.6406],
#         [372.5520, 198.2283, 390.2837, 247.0895],
#         [ 80.0485, 403.7431,  92.6021, 429.7381],
#         [ 82.1310, 383.9276, 103.3108, 430.0011],
#         [ 92.7740, 396.4763, 110.1527, 427.4432],
#         [  7.6112, 347.1008,  21.4179, 370.2100],
#         [106.4894, 408.8368, 124.7521, 429.9010],
#         [ 92.6113, 389.7758, 103.1128, 409.4579],
#         [ 91.8788, 390.7726, 102.9821, 406.6871],
#         [ 55.5639, 339.0872,  66.5498, 371.2108],
#         [299.1935, 333.7727, 332.4057, 367.5944],
#         [ 42.3521, 366.7964, 127.3372, 388.8732],
#         [298.2795, 161.7747, 322.8933, 181.2688],
#         [373.4181, 195.1609, 398.5721, 260.0147],
#         [116.7646, 377.8820, 126.7992, 396.0792],
#         [424.6078, 181.7227, 437.8418, 209.1708],
#         [ 85.5497, 406.8248, 102.3941, 429.9725],
#         [451.1175, 202.2093, 463.5256, 218.5361],
#         [ 52.4844, 349.7574,  60.8358, 369.1055],
#         [190.7394, 312.6267, 209.7696, 346.6253],
#         [185.4041, 312.5324, 224.1421, 371.7430],
#         [106.2660, 396.5806, 119.2361, 428.2131],
#         [193.3726, 310.7077, 234.0215, 349.4104],
#         [ 91.9059, 391.1800, 103.0363, 406.6176],
#         [344.7061, 139.8843, 378.1363, 186.3423]]), 'labels': tensor([ 6,  6,  1,  1,  1, 16,  1,  1,  6,  1,  1,  3,  3,  1,  1,  1,  8,  1,
#          2,  1,  1,  1,  2,  1, 31,  1,  1, 15,  1,  1,  1,  1,  2,  1,  1,  1,
#          1,  1,  1, 27,  1]), 'scores': tensor([0.9995, 0.9960, 0.9867, 0.9796, 0.9795, 0.9186, 0.9138, 0.8774, 0.8627,
#         0.8364, 0.7898, 0.6204, 0.5602, 0.3827, 0.3172, 0.3171, 0.2739, 0.2582,
#         0.2556, 0.2358, 0.2235, 0.2009, 0.1880, 0.1847, 0.1798, 0.1664, 0.1624,
#         0.1593, 0.1380, 0.1304, 0.1173, 0.1102, 0.1084, 0.0935, 0.0805, 0.0762,
#         0.0747, 0.0720, 0.0699, 0.0521, 0.0517])}]



# outputs: [{'boxes': tensor([[  0.0000,   0.0000,   0.0000,   0.0000],
#         [  0.0000,   0.0000,   0.0000,   0.0000],
#         [597.3008,   0.0000, 605.9858,   0.0000],
#         [  0.0000, 427.0000, 640.0000, 427.0000],
#         [  0.0000,   0.0000,   0.0000, 427.0000],
#         [  0.0000,   0.0000,   0.0000, 427.0000],
#         [  0.0000,   0.0000,   0.0000, 427.0000],
#         [482.4534,   0.0000, 482.4534, 427.0000],
#         [440.6744, 339.5648, 440.6744, 339.5648],
#         [555.1725, 279.4143, 555.1725, 279.4143],
#         [397.3072,   0.0000, 398.6817, 427.0000],
#         [640.0000,   0.0000, 640.0000,   0.0000]]), 'scores': tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9984, 0.9789,
#         0.0891, 0.0334, 0.0261]), 'labels': tensor([37, 37, 78, 78, 85, 85, 85, 62, 44, 84, 62, 62])}]