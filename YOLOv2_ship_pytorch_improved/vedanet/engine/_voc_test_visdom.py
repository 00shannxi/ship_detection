import logging as log
import torch
from torchvision import transforms as tf
from statistics import mean
import os
import cv2
from .. import data as vn_data
from .. import models
from . import engine
from utils.test import voc_wrapper

__all__ = ['VOCTest_visdom']
labelmap = (  # always index 0
    'bulk cargo carrier',
    'container ship',
    'fishing boat',
    'general cargo ship',
    'ore carrier',
    'passenger ship')
colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229),(158, 218, 229),(158, 218, 229)]
class CustomDataset(vn_data.BramboxDataset):
    def __init__(self, hyper_params):
        anno = hyper_params.testfile
        root = hyper_params.data_root
        network_size = hyper_params.network_size
        labels = hyper_params.labels


        lb  = vn_data.transform.Letterbox(network_size)
        it  = tf.ToTensor()
        img_tf = vn_data.transform.Compose([lb, it])
        anno_tf = vn_data.transform.Compose([lb])

        def identify(img_id):
            return f'{img_id}'

        super(CustomDataset, self).__init__('anno_pickle', anno, network_size, labels, identify, img_tf, anno_tf)

    def __getitem__(self, index):
        img, anno = super(CustomDataset, self).__getitem__(index)
        for a in anno:
            a.ignore = a.difficult  # Mark difficult annotations as ignore for pr metric
        return img, anno

def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

def VOCTest_visdom(hyper_params):
    log.debug('Creating network')

    model_name = hyper_params.model_name
    batch = hyper_params.batch
    use_cuda = hyper_params.cuda
    weights = hyper_params.weights
    conf_thresh = hyper_params.conf_thresh
    network_size = hyper_params.network_size
    labels = hyper_params.labels
    nworkers = hyper_params.nworkers
    pin_mem = hyper_params.pin_mem
    nms_thresh = hyper_params.nms_thresh
    #prefix = hyper_params.prefix
    results = hyper_params.results

    test_args = {'conf_thresh': conf_thresh, 'network_size': network_size, 'labels': labels}
    net = models.__dict__[model_name](hyper_params.classes, weights, train_flag=2, test_args=test_args)
    net.eval()
    log.info('Net structure\n%s' % net)
    #import pdb
    #pdb.set_trace()
    if use_cuda:
        net.cuda()

    log.debug('Creating dataset')
    loader = torch.utils.data.DataLoader(
        CustomDataset(hyper_params),
        batch_size = batch,
        shuffle = False,
        drop_last = False,
        num_workers = nworkers if use_cuda else 0,
        pin_memory = pin_mem if use_cuda else False,
        collate_fn = vn_data.list_collate,
    )

    log.debug('Running network')
    tot_loss = []
    coord_loss = []
    conf_loss = []
    cls_loss = []
    anno, det = {}, {}
    num_det = 0

    for idx, (data, box) in enumerate(loader):
        if (idx + 1) % 20 == 0:
            log.info('%d/%d' % (idx + 1, len(loader)))
        if use_cuda:
            data = data.cuda()
        img = loader.pull_image(idx)
        with torch.no_grad():
            detections, loss = net(data, box)

        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]])

        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.5:  # thresh 0.6
                # 因为在y = net(x)中Detect(num_classes, 0, 200, 0.01, 0.45)已经保留了200个最高分往下的框
                # 所以这里的0.6相当于eval中的ovthresh：Overlap threshold (default = 0.5)，来保留最终的框
                score = detections[0, i, j, 0]
                label_name = labelmap[i - 1]
                display_txt = '%s: %.2f' % (label_name, score)
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                # coords = (pt[0], pt[1], pt[2], pt[3])
                color = colors_tableau[i]
                cv2.rectangle(img, (pt[0], pt[1]), (pt[2], pt[3]), color, 2)  # 画框出物体的大框
                # print(pt)
                t_size = cv2.getTextSize(display_txt, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                c2 = pt[0] + t_size[0] + 3, pt[1] + t_size[1] + 4  # 下面一行分别对两个坐标取了int
                cv2.rectangle(img, (int(pt[0]), int(pt[1])), (int(pt[0] + t_size[0] + 3), int(pt[1] + t_size[1] + 4)),
                              color, -1)  # 画标签的框
                cv2.putText(img, display_txt, (int(pt[0]), int(pt[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0),
                            1, 8)
                # cv2.putText(img, str(i), (123,456)), font, 2, (0,255,0), 3)
                # 各参数依次是：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细,(0,0,0)表示黑色，白色是(255,255,255)
                j += 1
        cv2.imwrite('./show_ship/img_%d.jpg' % (idx), img)

    #     key_val = len(anno)
    #     anno.update({loader.dataset.keys[key_val+k]: v for k,v in enumerate(box)})
    #     det.update({loader.dataset.keys[key_val+k]: v for k,v in enumerate(output)})
    #
    # netw, neth = network_size
    # reorg_dets = voc_wrapper.reorgDetection(det, netw, neth) #, prefix)
    # voc_wrapper.genResults(reorg_dets, results, nms_thresh)


