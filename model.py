import timm
import torch.nn as nn

def get_model(num_classes=8, pretrained=True):
    model = timm.create_model('resnet18', pretrained=pretrained)
    # 修改最後 fc 層輸出數
    in_features = model.get_classifier().in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
