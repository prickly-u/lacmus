# -*- coding: utf-8 -*-
"""
Parameters for the encoders

Copyright 2019 IlyaDobrynin (https://github.com/IlyaDobrynin)

"""

resnet_layers = (
    ['conv1', 'bn1', 'relu'],
    ['maxpool', 'layer1'],
    ['layer2'],
    ['layer3'],
    ['layer4']
)

encoder_dict = {
    'resnet18': {
        'skip': resnet_layers,
        'filters': (64, 64, 128, 256, 512),
        'features': False
    },
    'resnet34': {
        'skip': resnet_layers,
        'filters': (64, 64, 128, 256, 512),
        'features': False
    },
    'resnet50': {
        'skip': resnet_layers,
        'filters': (64, 256, 512, 1024, 2048),
        'features': False
    },
    'resnet101': {
        'skip': resnet_layers,
        'filters': (64, 256, 512, 1024, 2048),
        'features': False
    },
    'resnet152': {
        'skip': resnet_layers,
        'filters': (64, 256, 512, 1024, 2048),
        'features': False
    },

    'resnext50': {
        'skip': resnet_layers,
        'filters': (64, 256, 512, 1024, 2048),
        'features': False
    },
    'resnext101': {
        'skip': resnet_layers,
        'filters': (64, 256, 512, 1024, 2048),
        'features': False
    },
    'se_resnext50': {
        'skip': resnet_layers,
        'filters': (64, 256, 512, 1024, 2048),
        'features': False
    }
}