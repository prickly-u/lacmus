# -*- coding: utf-8 -*-
"""
Module implements class which returns encoder autobuilder

Copyright 2019 IlyaDobrynin (https://github.com/IlyaDobrynin)

"""
import gc
from torch import nn
from nnets.models.backbones import backbone_factory
from nnets.models.backbones.pretrain_parameters import encoder_dict


class EncoderCommon(nn.Module):
    def __init__(self, backbone, pretrained, depth, unfreeze_encoder):
        super(EncoderCommon, self).__init__()

        assert backbone in backbone_factory.BACKBONES.keys(), \
            f"Wrong name of backbone: {backbone}. " \
                f"Should be in backbones.backbone_factory.backbones.keys()"
        
        self.depth = depth
        self.backbone = backbone
        self.encoder = backbone_factory.get_backbone(
            name=self.backbone,
            pretrained=pretrained,
            requires_grad=unfreeze_encoder
        )
        self.encoder_layers_dict = encoder_dict[backbone]['skip']
        self.encoder_filters = encoder_dict[backbone]['filters']
        self.is_featured = encoder_dict[backbone]['features']
        self.encoder_layers = self._get_encoder()
    
    def _get_encoder(self):
        """ Function to define u-net encoder layers

        :return: List of encoder layers
        """
        encoder_list = nn.ModuleList([])
        if self.is_featured:
            for (mk, mv) in self.encoder.named_children():
                if mk == 'features':
                    for i in range(self.depth):
                        encoder_layer = nn.ModuleList([])
                        for layer in self.encoder_layers_dict[i]:
                            encoder_layer.append(dict(mv.named_children())[layer])
                        encoder_list.append(nn.Sequential(*encoder_layer))
                else:
                    continue
        else:
            for i in range(self.depth):
                encoder_layer = nn.ModuleList([])
                for layer in self.encoder_layers_dict[i]:
                    encoder_layer.append(dict(self.encoder.named_children())[layer])
                encoder_list.append(nn.Sequential(*encoder_layer))
        del self.encoder
        gc.collect()
        return encoder_list
    
    def _make_encoder_forward(self, x):
        """ Function to make u-net encoder

        :param x: Input tenzor
        :return: List of encoder tensors
        """
        encoder_list = []
        if self.backbone in ['pnasnet5large', 'nasnetalarge']:
            encoder_list_tmp = []
            counter = 2
            for i, outer_layer in enumerate(self.encoder_layers):
                if i < 2:
                    x = outer_layer(x)
                    encoder_list.append(x.clone())
                    encoder_list_tmp.append(x.clone())
                    continue
                else:
                    for inner_layer in outer_layer:
                        if self.backbone == 'nasnetalarge':
                            first_layer = encoder_list_tmp[counter - 1]
                            if counter == 2:
                                first_layer = encoder_list_tmp[counter - 2]
                                second_layer = encoder_list_tmp[counter - 1]
                            elif counter in (10, 17):
                                second_layer = encoder_list_tmp[counter - 3]
                            else:
                                second_layer = encoder_list_tmp[counter - 2]
                        else:
                            first_layer = encoder_list_tmp[counter - 2]
                            second_layer = encoder_list_tmp[counter - 1]
                        x = inner_layer(first_layer, second_layer)
                        encoder_list_tmp.append(x.clone())
                        counter += 1
                    encoder_list.append(x.clone())
            del encoder_list_tmp
            gc.collect()
        else:
            for encoder_layer in self.encoder_layers:
                x = encoder_layer(x)
                encoder_list.append(x.clone())
        return x, encoder_list
    
    def forward(self, x):
        raise NotImplementedError(
            f"Method forward at encoder is not implemented"
        )

