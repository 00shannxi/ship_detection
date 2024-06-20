"""Reval = re-eval. Re-evaluate saved detections.

usage:
        input with the command: $python reval_voc.py --voc_dir VOCdevkit --year 2007 --image_set test --class ./data/voc.names
                            Actually we input $ python reval_voc.py --voc_dir C:\\Users\\Breeze\\Desktop\\
                                                    Mask-or-Not\\darknet\\build\\darknet\\x64\\results
                            will be okay,since I have got the default value changed to my path.
                            注释里面的\得换成\\否则会报错
NOTE:this .py has to be opened with the results.file,otherwise the import of voc_eval would break with error
"""
#实现计算p r  AP mAP
import argparse
import os
import pickle as cPickle
import sys

import numpy as np

# from voc_eval import voc_eval
import xml.etree.ElementTree as ET
import os
# import pickle as cPickle
# import numpy as np
# def parse_args():
#     """
#     Parse input arguments
#     """
#     #以下几个argument，我在原来基础上都加了default，其实看着我改过的就很容易理解
#     #voc_dir就是VOCdevkit的路径
#     #year默认成你文件夹对应的，我是2020
# 	#几个可选变量都设成了默认，所以在cmd就只需要输入必选变量output_dir 即可，也就是生成文件保存在这个地#方
#     parser = argparse.ArgumentParser(description='Re-evaluate results')
#     # parser.add_argument('output_dir', nargs=1, help='results directory',
#     #                     type=str)
#     # parser.add_argument('--voc_dir', dest='voc_dir', default='F:/pycharm/YOLOv2_ship_pytorch/data/Seaships_7000/VOCdevkit', type=str)
#     # parser.add_argument('--year', dest='year', default='2020', type=str)
#     parser.add_argument('--image_set', dest='image_set', default='test', type=str)
#
#     # parser.add_argument('--classes', dest='class_file', default='C:\\Users\\Breeze\\Desktop\\darknet\\'
#     #                                                             'darknet-master\\build\darknet\\x64\\data\\voc.names',
#     #                      type=str)
#
#     if len(sys.argv) == 1:
#         parser.print_help()
#         sys.exit(1)
#
#     args = parser.parse_args()
#     return args
image_set = 'test'
devkit_PR_path = './'

classes = (  # always index 0
    'bulk cargo carrier',
    'container ship',
    'fishing boat',
    'general cargo ship',
    'ore carrier',
    'passenger ship')

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        #obj_struct['pose'] = obj.find('pose').text
        #obj_struct['truncated'] = int(obj.find('truncated').text)
        # obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.45,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)

    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots

        recs = {}
        for i, imagename in enumerate(imagenames):
            #print(annopath.format(imagename))
            #print(imagenames)
            recs[imagename] = parse_rec("F:/pycharm/YOLOv2_ship_pytorch_improved/data/Seaships_7000/VOCdevkit/VOC2007"
                                        "/Annotations/"+annopath.format(imagename))
            #print("PY")
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            #print(cachefile)
            recs = cPickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        # difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        # npos = npos + sum(~difficult)
        npos = npos + len(R)
        class_recs[imagename] = {'bbox': bbox,
                                 # 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    #原文用的相对路径，不是太好控制，所以这里直接改成绝对路径
    detfile="F:\pycharm\YOLOv2_ship_pytorch_improved\\results\\"+detfile

    #print(detfile)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        # if ovmax > ovthresh:
        #     if not R['difficult'][jmax]:
        #         if not R['det'][jmax]:
        #             tp[d] = 1.
        #             R['det'][jmax] = 1
        #         else:
        #             fp[d] = 1.
        # else:
        #     fp[d] = 1.

        if ovmax > ovthresh:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

def get_voc_results_file_template(image_set, out_dir='results'):  #读取test.py生成的检测结果.txt文件
    filename = 'comp4_det_' + image_set + '_{:s}.txt'
    path = os.path.join(out_dir, filename)
    return path

def get_voc_P_results_file_template(cls):#  P 结果保存的路径以及文件名
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + 'P' + '_yolov2_improved'+ '_%s.txt' % (cls)
    filedir = os.path.join(devkit_PR_path, 'P_results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path

def get_voc_R_results_file_template(cls):#  R  结果保存的路径以及文件名
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + 'R' + '_yolov2_improved'+ '_%s.txt' % (cls)
    filedir = os.path.join(devkit_PR_path, 'R_results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def do_python_eval(devkit_path, image_set, classes, output_dir='results_eval'):
    annopath = os.path.join(
        devkit_path,
        'VOC2007',
        'Annotations',
        '{:s}.xml')
    imagesetfile = os.path.join(
        devkit_path,
        'VOC2007',
        'ImageSets',
        'Main',
        image_set + '.txt')
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    # use_07_metric = True if int(year) < 2010 else False
    # print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    use_07_metric = True
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for i, cls in enumerate(classes):
        # if cls == '__background__':
        #     continue

        filename = get_voc_results_file_template(image_set).format(cls)
        rec, prec, ap = voc_eval(
            filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.45,
            use_07_metric=use_07_metric)
        print("HERE")
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))

        filename_P = get_voc_P_results_file_template(cls)
        filename_R = get_voc_R_results_file_template(cls)
        with open(filename_R, 'a') as f:  # 要将一行多列的数组存入txt,就需要添加下面这一行的for循环
            for mr in rec:
                f.write(str('%.3f' % (mr)) + '\n')
        # f.write(str('%.3f' % (rec.item())) + '\n')
        with open(filename_P, 'a') as f:  # 保存到txt,用matlab画图
            for mp in prec:
                f.write(str('%.3f' % (mp)) + '\n')

        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('-- Thanks, The Management')
    print('--------------------------------------------------------------')


if __name__ == '__main__':
    # args = parse_args()
    voc_dir = 'F:/pycharm/YOLOv2_ship_pytorch_improved/data/Seaships_7000/VOCdevkit'
    # output_dir = os.path.abspath(args.output_dir[0])
    # print(args.class_file)
    # with open(args.class_file, 'r') as f:
    #     lines = f.readlines()
    #
    # classes = [t.strip('\n') for t in lines]

    print('Evaluating detections')
    # do_python_eval(args.voc_dir, args.image_set, classes, output_dir)
    do_python_eval(voc_dir, image_set, classes, output_dir='results_eval')