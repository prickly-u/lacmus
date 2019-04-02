# -*- coding: utf-8 -*-
"""
Module implements classification auto builder class

Copyright 2019 IlyaDobrynin (https://github.com/IlyaDobrynin)

"""
import torch
from torch import nn
from nnets.models.nn_blocks.encoders import EncoderCommon


class ClassifierFactory(EncoderCommon):
    """
    Classification auto builder class
    
    Arguments:
        backbone:           name of the u-net encoder line.
                            Should be in backbones.backbone_factory.backbones.keys()
        num_classes:        amount of output classes to predict
        pretrained:         name of the pretrain weights. 'imagenet' or None
        unfreeze_encoder:   Flag to unfreeze encoder weights
    """
    def __init__(self, backbone, num_classes, pretrained='imagenet', unfreeze_encoder=True):

        super(ClassifierFactory, self).__init__(backbone=backbone,
                                                pretrained=pretrained,
                                                depth=5,
                                                unfreeze_encoder=unfreeze_encoder)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(self.encoder_filters[-1], num_classes)

    def forward(self, x):
        x, _ = self._make_encoder_forward(x)
        out = self.avgpool(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def test_class(backbone_name):
    """ Testing method
    
    :param backbone_name:
    :return:
    """
    try:
        input_size = (3, 256, 256)
        model = ClassifierFactory(
            backbone=backbone_name, num_classes=1, pretrained='imagenet', unfreeze_encoder=True
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        from torchsummary import summary
        summary(model, input_size=input_size)
    except Exception as e:
        raise Exception(
            f"Classification autobuilder exception: {repr(e)}"
        )


if __name__ == '__main__':
    test_class(backbone_name='resnet34')
    

