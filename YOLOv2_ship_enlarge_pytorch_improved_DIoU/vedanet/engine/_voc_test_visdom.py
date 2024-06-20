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
import numpy as np
from PIL import Image

__all__ = ['VOCTest_visdom']
label_names = ( 'bulk cargo carrier','container ship','fishing boat',
    'general cargo ship','ore carrier','passenger ship')
#label_names中的编号从0到5
# colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
#                  (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
#                  (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
#                  (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
#                  (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229),(158, 218, 229),(158, 218, 229)]
#表示24种颜色
colors_tableau = [(220, 20, 60), (0, 255, 127), (255, 255, 0), (0, 255, 255), (192, 14, 235),
                 (0, 0, 255)]  #红  绿 黄  青 棕红 蓝

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

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
    # anno, det = {}, {}   #空字典
    num_det = 0
    anno, det = {}, {}  # 空字典
    for idx, (data, box) in enumerate(loader):   #idx表示一个batch
        if (idx + 1) % 20 == 0:
            log.info('%d/%d' % (idx + 1, len(loader)))
        if use_cuda:
            data = data.cuda()
        with torch.no_grad():
            output, loss = net(data, box)
        # print(output)
        key_val = len(anno)
        anno.update({loader.dataset.keys[key_val+k]: v for k,v in enumerate(box)})
        det.update({loader.dataset.keys[key_val+k]: v for k,v in enumerate(output)})  #保存的是batch里所有图的预测框  比如batch为2，那就是2幅图的预测结果
        # print(det)
#{'F:\\pycharm\\YOLOv2_ship_pytorch\\data\\Seaships_7000\\VOCdevkit\\VOC2007\\JPEGImages\\000013.jpg': [Detection {class_label = fishing boat, object_id = 0, x
#= 102.21758270263672, y = 241.95773315429688, w = 19.921459197998047, h = 8.139739036560059, confidence = 0.042907655239105225}, Detection {class_label = passe
#nger ship, object_id = 0, x = 102.21758270263672, y = 241.95773315429688, w = 19.921459197998047, h = 8.139739036560059, confidence = 0.06380932778120041}, Det
#ection {class_label = fishing boat, object_id = 0, x = 233.664306640625, y = 231.80691528320312, w = 7.328129768371582, h = 4.918141841888428, confidence = 0.0
#19558332860469818}]}
        netw, neth = network_size
    # reorg_dets = voc_wrapper.reorgDetection(det, netw, neth) #, prefix)
    # voc_wrapper.genResults(reorg_dets, results, nms_thresh)
        for k, v in det.items():  #遍历字典   k为图片  v为检测结果   遍历一个batch里的16张图  一次遍历一幅图
            # img_fp = '%s/%s.jpg' % (prefix, k)
            # print('a')
            # print(v)
            img_fp = k  # '%s/%s.jpg' % (prefix, k)   K表示一幅图
            # name = k.split('/')[-1]
            # name = k.split('/')[-1][
            #        :-4]  # F:\pycharm\YOLOv2_ship_pytorch\data\Seaships_7000\VOCdevkit\VOC2007\JPEGImages\006971
            name = k[52:-4]  #52表示E:\paper6\Seaships_new\VOCdevkit\VOC2007\JPEGImages\的字符数，-4表示去掉.jpg
            #E:\paper6\Seaships_new\VOCdevkit\VOC2007\JPEGImages\006971.jpg
            fd = cv2.imread(img_fp, cv2.IMREAD_COLOR)
            # with Image.open(img_fp) as fd:
            # orig_width, orig_height = fd.size  # 原始的宽和高
            imgInfo = fd.shape  # 原始的宽和高
            orig_width = imgInfo[1]
            orig_height = imgInfo[0]
            scale = min(float(netw) / orig_width, float(neth) / orig_height)
            new_width = orig_width * scale
            new_height = orig_height * scale
            pad_w = (netw - new_width) / 2.0
            pad_h = (neth - new_height) / 2.0
            # img = cv2.cvtColor(np.asarray(fd), cv2.COLOR_RGB2BGR)
            pt1 = []
            pt2 = []
            pt3 = []
            pt4 = []
            for iv in v:    #遍历一张图上的所有框
                # print('a')
                # print(iv)
                xmin = iv.x_top_left
                ymin = iv.y_top_left
                xmax = xmin + iv.width
                ymax = ymin + iv.height
                conf = iv.confidence
                # print(conf)
                class_label = iv.class_label
                # print(xmin, ymin, xmax, ymax)

                xmin = max(0, float(xmin - pad_w) / scale)  # 计算出来的是到原图上的具体位置
                xmax = min(orig_width - 1, float(xmax - pad_w) / scale)
                ymin = max(0, float(ymin - pad_h) / scale)
                ymax = min(orig_height - 1, float(ymax - pad_h) / scale)

                # reorg_dets.setdefault(class_label, {})
                # reorg_dets[class_label].setdefault(name, [])
                # line = '%s %f %f %f %f %f' % (name, conf, xmin, ymin, xmax, ymax)
                pt = [xmin, ymin, xmax, ymax, conf]   #这个值是经过上面的转换后对应到原图上的坐标   但是是浮点型
                # print(pt)
                pt1.append(pt)   #把所有框都放到pt1中[[],[],...,[]]   pt1是个list
                # pt2 = np.array(pt1)
                # print(pt2)
            # print(pt2)
            # keep = py_cpu_nms(pt2, 0.5)
            for pt2 in pt1:   #遍历所有的框
                if pt2[4] >= 0.05:  #置信度阈值大于0.05
                    pt3.append(pt2)  #存放到一个list中
                    pt4 = np.array(pt3)  #转为ndarry才能调用nms
            # if pt2[4] >= 0.2:  #置信度阈值大于0.5
            keep = py_cpu_nms(pt4, 0.5)  #得到的是框的索引  0.5  0.45
                    # pt_new = pt3
            print(keep)
            for i in keep:  #遍历索引  找框
                pt_new = pt4[i]  #根据索引找到框
                display_txt = '%s: %.2f' % (class_label, pt_new[4])
                color = colors_tableau[label_names.index(class_label)]   #根据类别名，找到在label_names中的序号，从0开始
                cv2.rectangle(fd, (int(pt_new[0]), int(pt_new[1])), (int(pt_new[2]), int(pt_new[3])), color, 3)   #画框目标的大框，一定要用整型，用浮点会报错，说只输入了两个参数，因为没有读取浮点数。
                t_size = cv2.getTextSize(display_txt, cv2.FONT_HERSHEY_PLAIN, 2, 1)[0]
                    # c2 = pt_new[0] + t_size[0] + 3, pt_new[1] + t_size[1] + 4  # 下面一行分别对两个坐标取了int
                    # cv2.rectangle(fd, (int(pt_new[0]), int(pt_new[1])),
                    #               (int(pt_new[0] + t_size[0] + 3), int(pt_new[1] + t_size[1] + 4)),
                    #               color, -1)  # 画标签的框
                cv2.rectangle(fd,(int(pt_new[0]), int(pt_new[1]-40)), (int(pt_new[0] + t_size[0]+175), int(pt_new[1])),
                                  color, -1)  # 画标签的框
                cv2.putText(fd, display_txt, (int(pt_new[0]), int(pt_new[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                (0, 0, 0), 2)#显示类别和置信度  黑色的字  字号  线的粗细
            cv2.imwrite('F:\pycharm\YOLOv2_ship_enlarge_pytorch_improved_DIoU\show_ship\%s.jpg' % (name), fd)