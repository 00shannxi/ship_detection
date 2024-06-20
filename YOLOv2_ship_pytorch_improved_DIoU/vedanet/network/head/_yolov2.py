import os
from collections import OrderedDict
import torch
import torch.nn as nn

from .. import layer as vn_layer

__all__ = ['Yolov2']


class Yolov2(nn.Module):
    def __init__(self, num_classes):
        """ Network initialisation """
        super().__init__()
        layer_list = [
            # Sequence 2 : input = sequence0
            OrderedDict([
                ('1_convbatch',    vn_layer.Conv2dBatchLeaky(512, 64, 1, 1)),#26 × 26 × 64
                ('2_reorg',        vn_layer.Reorg(2)),  #13×13×256
                ]),

            # Sequence 3 : input = sequence2 + sequence1
            OrderedDict([
                ('3_convbatch',    vn_layer.Conv2dBatchLeaky((4*64)+1024, 1024, 3, 1)),   #13 × 13 × 1024
                ('4_conv',         nn.Conv2d(1024, 3*(5+num_classes), 1, 1, 0)),#13 × 13 × 33
                ]),

            OrderedDict([
                ('5_convbatch', vn_layer.Conv2dBatchLeaky(256, 64, 1, 1)),  # 52 × 52 × 64
                ('2_reorg', vn_layer.Reorg(2)),  # 26×26×64*4
            ]),
            OrderedDict([
                ('6_convbatch', vn_layer.Conv2dBatchLeaky(1024, 512, 1, 1)),  # 13 × 13 × 512
                ('7_convbatch', nn.Upsample(scale_factor=2)),  # 26×26×512
            ]),

            # Sequence 3 : input = sequence2 + sequence1
            OrderedDict([
                # ('7_convbatch', vn_layer.Conv2dBatchLeaky((4 * 256) + 512 + 512, 1024, 3, 1)),  # 26 × 26 × 1024
                ('7_convbatch', vn_layer.Conv2dBatchLeaky( (4 * 64)+512 + 512, 1024, 3, 1)),  # 26 × 26 × 1280
                ('8_conv', nn.Conv2d(1024, 2 * (5 + num_classes), 1, 1, 0)),  # 26 × 26 × 22
            ]),
            ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

        # layer_list_1 = [
        #     # Sequence 2 : input = sequence0
        #     OrderedDict([
        #         # ('1_convbatch', vn_layer.Conv2dBatchLeaky(512, 64, 1, 1)),  # 26 × 26 × 64
        #         ('1_reorg', vn_layer.Reorg(2)),  # 26×26×256*4
        #     ]),
        #     OrderedDict([
        #         ('2_convbatch',vn_layer.Conv2dBatchLeaky(1024, 512, 1, 1)),  # 13 × 13 × 512
        #         ('3_convbatch', nn.Upsample(scale_factor=2)),  # 26×26×512
        #     ]),
        #
        #     # Sequence 3 : input = sequence2 + sequence1
        #     OrderedDict([
        #         ('4_convbatch', vn_layer.Conv2dBatchLeaky((4 * 256) + 512 + 512, 1024, 3, 1)),  # 26 × 26 × 1024
        #         ('5_conv', nn.Conv2d(1024, 2*(5 + num_classes), 1, 1, 0)),  # 26 × 26 × 22
        #     ]),
        # ]
        # self.layers_1 = nn.ModuleList([nn.Sequential(layer_dict_1) for layer_dict_1 in layer_list_1])

    def forward(self, middle_feats):
        outputs = []
        # stage 5
        # Route : layers=-9
        stage6_reorg = self.layers[0](middle_feats[1])   #13×13×256
        # stage 6
        stage6 = middle_feats[0]  #13 × 13 × 1024
        # Route : layers=-1, -4
        out = self.layers[1](torch.cat((stage6_reorg, stage6), 1))   #圆括号里边的是输入
        # features = [out]    #13 × 13 × 55
        # stage7_reorg = self.layers_1[0](middle_feats[2])  #26×26×1024
        stage7_reorg = self.layers[2](middle_feats[2])  # 26×26×256
        stage8 = middle_feats[1] ##26×26×512
        # stage9 = self.layers_1[1](middle_feats[0])  #26×26×512
        stage9 = self.layers[3](middle_feats[0])  # 26×26×512
        # out_26 = self.layers_1[2](torch.cat((stage7_reorg, stage8, stage9), 1))
        out_26 = self.layers[4](torch.cat((stage7_reorg, stage8, stage9), 1))
        features = [out,out_26] #13 × 13 × 33,26 × 26 × 22
        return features
