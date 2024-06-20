import os
from collections import OrderedDict, Iterable
import torch
import torch.nn as nn
from .. import loss
from .yolo_abc import YoloABC
from ..network import backbone
from ..network import head

__all__ = ['Yolov2']

#anchors = [(0.982, 0.457), (2.085, 0.831),(3.683, 1.396), (6.371, 1.998), (8.849, 3.298)],
class Yolov2(YoloABC):
    def __init__(self, num_classes=6, weights_file=None, input_channels=3,
                 anchors = [(31.424, 14.624), (66.72, 26.592),(117.856, 44.672), (203.872, 63.936), (283.168, 105.536)],
                 anchors_mask=[(2,3,4),(0,1)], train_flag=1, clear=False, test_args=None):
        """ Network initialisation """
        super().__init__()

        # Parameters
        self.num_classes = num_classes
        self.anchors = anchors
        self.anchors_mask = anchors_mask
        self.nloss = len(self.anchors_mask)   #5
        self.train_flag = train_flag
        self.test_args = test_args

        self.loss = None
        self.postprocess = None

        self.backbone = backbone.Darknet19()
        self.head = head.Yolov2(num_classes=num_classes)

        if weights_file is not None:
            self.load_weights(weights_file, clear)
        else:
            self.init_weights(slope=0.1)

    def _forward(self, x):
        middle_feats = self.backbone(x)   #features = [stage6, stage5, stage4]
        features = self.head(middle_feats)
        loss_fn = loss.RegionLoss
        # loss_fn = loss.YoloLoss
        
        self.compose(x, features, loss_fn)

        return features

    def modules_recurse(self, mod=None):
        """ This function will recursively loop over all module children.

        Args:
            mod (torch.nn.Module, optional): Module to loop over; Default **self**
        """
        if mod is None:
            mod = self

        for module in mod.children():
            if isinstance(module, (nn.ModuleList, nn.Sequential, backbone.Darknet19, head.Yolov2)):
                yield from self.modules_recurse(module)
            else:
                yield module
