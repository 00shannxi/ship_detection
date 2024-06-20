import pickle
import numpy as np
from PIL import Image
# from .fast_rcnn.nms_wrapper import nms, soft_nms
# from .nms.nms_wrapper import nms, soft_nms
from .nms.py_cpu_nms import py_cpu_nms
import os

devkit_PR_path = './VOCPR/'
def genResults(reorg_dets, results_folder, nms_thresh=0.45):
    for label, pieces in reorg_dets.items():
        ret = []
        dst_fp = '%s/comp4_det_test_%s.txt' % (results_folder, label)
        for name in pieces.keys():
            pred = np.array(pieces[name], dtype=np.float32)
            # keep = nms(pred, nms_thresh, force_cpu=False)  #GPU cython
            # keep = nms(pred, nms_thresh, force_cpu=True)#cpu cython
            #keep = soft_nms(pred, sigma=0.5, Nt=0.3, method=1)
            keep = py_cpu_nms(pred, nms_thresh) #纯python
            #print k, len(keep), len(pred_dets[k])
            for ik in keep:
                #print k, pred_left[ik][-1], ' '.join([str(int(num)) for num in pred_left[ik][:4]])
                line ='%s %f %s' % (name, pred[ik][-1], ' '.join([str(num) for num in pred[ik][:4]]))
                # ‘’。join()连接字符串数组。将字符串、元组、列表中的元素以指定的字符(分隔符)连接生成一个新的字符串
                ret.append(line)

        with open(dst_fp, 'w') as fd:  #打开txt文件
            fd.write('\n'.join(ret))   #将文件名 置信度 四个坐标存到txt中
            #F:\pycharm\YOLOv2_ship_pytorch\data\Seaships_7000\VOCdevkit\VOC2007\JPEGImages\006971 0.020793 1830.2283 530.14386 1907.9525 572.1158

#得到预测的所有的位置坐标和置信度
def reorgDetection(dets, netw, neth): #, prefix):     #netw  neth 为输入的测试图像尺寸  416 416
    reorg_dets = {}
    for k, v in dets.items():
        #img_fp = '%s/%s.jpg' % (prefix, k)
        img_fp = k #'%s/%s.jpg' % (prefix, k)
        #name = k.split('/')[-1]
        # name = k.split('/')[-1][:-4]   #E:\paper6\Seaships_new\VOCdevkit\VOC2007\JPEGImages\006971
        name = k[52:-4]  #E:\paper6\Seaships_new\VOCdevkit\VOC2007\JPEGImages\
        with Image.open(img_fp) as fd:
            orig_width, orig_height = fd.size    #原始的宽和高
        scale = min(float(netw)/orig_width, float(neth)/orig_height)
        new_width = orig_width * scale
        new_height = orig_height * scale
        pad_w = (netw - new_width) / 2.0
        pad_h = (neth - new_height) / 2.0

        for iv in v:
            xmin = iv.x_top_left
            ymin = iv.y_top_left
            xmax = xmin + iv.width
            ymax = ymin + iv.height
            conf = iv.confidence
            class_label = iv.class_label
            #print(xmin, ymin, xmax, ymax)

            xmin = max(0, float(xmin - pad_w)/scale)  #计算出来的是到原图上的具体位置
            xmax = min(orig_width - 1,float(xmax - pad_w)/scale)
            ymin = max(0, float(ymin - pad_h)/scale)
            ymax = min(orig_height - 1, float(ymax - pad_h)/scale)

            reorg_dets.setdefault(class_label, {})
            reorg_dets[class_label].setdefault(name, [])
            #line = '%s %f %f %f %f %f' % (name, conf, xmin, ymin, xmax, ymax)
            piece = (xmin, ymin, xmax, ymax, conf)
            reorg_dets[class_label][name].append(piece)

    return reorg_dets





def main():
    netw, neth = 416, 416
    results_folder = 'results_test'
    prefix = '/data/Seaships_7000/VOCdevkit'
    with open('yolov2_bilinear_85000_416_bilinear.pkl', 'rb') as fd:   #yolov2_bilinear_85000_416_bilinear.pkl
        dets = pickle.load(fd)
    reorg_dets = reorgDetection(dets, netw, neth, prefix)
    genResults(reorg_dets, results_folder)


if __name__ == '__main__':
    main()
