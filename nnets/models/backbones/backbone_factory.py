# -*- coding: utf-8 -*-
"""
Simple backbones factory

Copyright 2019 IlyaDobrynin (https://github.com/IlyaDobrynin)

"""
from src.models.backbones.pretrained_models import (
    resnet101, resnet50, resnet34, resnet18
    
)



BACKBONES = {
    # Done
    'resnet101': resnet101,                      # 77.438	93.672
    'resnet50': resnet50,                        # 76.002	92.980
    'resnet34': resnet34,                        # 73.554	91.456
    'resnet18': resnet18,                        # 70.142	89.274
}


def get_backbone(name, *args, **kwargs):
    """ Function returns pytorch pretrained model with given args and kwargs
    from the list of backbones

    :param name: Pretrained backbone name
    :param args: Model arguments
    :param kwargs: Model keyword arguments
    :return: Model
    """
    return BACKBONES[name](*args, **kwargs)
