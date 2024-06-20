import torch
import math
def bbox_ious(boxes1, boxes2):
    """ Compute IOU between all boxes from ``boxes1`` with all boxes from ``boxes2``.

    Args:
        boxes1 (torch.Tensor): List of bounding boxes
        boxes2 (torch.Tensor): List of bounding boxes

    Note:
        List format: [[xc, yc, w, h],...]
    """
    b1_len = boxes1.size(0)
    b2_len = boxes2.size(0)

    b1x1, b1y1 = (boxes1[:, :2] - (boxes1[:, 2:4] / 2)).split(1, 1)
    b1x2, b1y2 = (boxes1[:, :2] + (boxes1[:, 2:4] / 2)).split(1, 1)
    b2x1, b2y1 = (boxes2[:, :2] - (boxes2[:, 2:4] / 2)).split(1, 1)
    b2x2, b2y2 = (boxes2[:, :2] + (boxes2[:, 2:4] / 2)).split(1, 1)

    dx = (b1x2.min(b2x2.t()) - b1x1.max(b2x1.t())).clamp(min=0)
    dy = (b1y2.min(b2y2.t()) - b1y1.max(b2y1.t())).clamp(min=0)
    intersections = dx * dy

    areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)
    unions = (areas1 + areas2.t()) - intersections

    return intersections / unions

def bbox_overlaps_iou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = torch.zeros((cols, rows))
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
        bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
        bboxes2[:, 3] - bboxes2[:, 1])

    inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    union = area1+area2-inter_area
    ious = inter_area / union
    ious = torch.clamp(ious,min=0,max = 1.0)
    if exchange:
        ious = ious.T
    return ious

def bbox_diou_1(bboxes1, bboxes2):   #原版
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    dious = torch.zeros((rows, cols))
    if rows * cols == 0:  #
        return dious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        dious = torch.zeros((cols, rows))
        exchange = True
    # #xmin,ymin,xmax,ymax->[:,0],[:,1],[:,2],[:,3]
    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = ((bboxes1[:, 2] + bboxes1[:, 0]) / 2) #（x1max +x1min）/2
    center_y1 = ((bboxes1[:, 3] + bboxes1[:, 1]) / 2)  #(y1max+y1min)/2
    center_x2 = ((bboxes2[:, 2] + bboxes2[:, 0]) / 2)
    center_y2 = ((bboxes2[:, 3] + bboxes2[:, 1]) / 2)

    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:]) #min((x1max,y1max ),(x2max,y2max)) ->返回较小一组
    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2]) ##max((x1min,y1min ),(x2min,y2min))->返回较大的一组
    out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1 + area2 - inter_area
    dious = inter_area / union - (inter_diag) / outer_diag
    dious = torch.clamp(dious, min=-1.0, max=1.0)
    if exchange:
        dious = dious.T
    return dious

def bbox_diou(bboxes1, bboxes2):
    # rows = bboxes1.shape[0]
    # cols = bboxes2.shape[0]
    # dious = torch.zeros((rows, cols))
    # if rows * cols == 0:  #
    #     return dious
    # exchange = False
    # if bboxes1.shape[0] > bboxes2.shape[0]:
    #     bboxes1, bboxes2 = bboxes2, bboxes1
    #     dious = torch.zeros((cols, rows))
    #     exchange = True
    # #xmin,ymin,xmax,ymax->[:,0],[:,1],[:,2],[:,3]
    b1x1, b1y1 = (bboxes1[:, :2] - (bboxes1[:, 2:4] / 2)).split(1, 1)
    b1x2, b1y2 = (bboxes1[:, :2] + (bboxes1[:, 2:4] / 2)).split(1, 1)
    b2x1, b2y1 = (bboxes2[:, :2] - (bboxes2[:, 2:4] / 2)).split(1, 1)
    b2x2, b2y2 = (bboxes2[:, :2] + (bboxes2[:, 2:4] / 2)).split(1, 1)
    dx = (b1x2.min(b2x2.t()) - b1x1.max(b2x1.t())).clamp(min=0)
    dy = (b1y2.min(b2y2.t()) - b1y1.max(b2y1.t())).clamp(min=0)
    inter_area = dx * dy

    areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)
    union = (areas1 + areas2.t()) - inter_area



    # w1 = bboxes1[:, 2] - bboxes1[:, 0]
    # h1 = bboxes1[:, 3] - bboxes1[:, 1]
    # w2 = bboxes2[:, 2] - bboxes2[:, 0]
    # h2 = bboxes2[:, 3] - bboxes2[:, 1]

    # area1 = w1 * h1
    # area2 = w2 * h2

    # center_x1 = ((bboxes1[:, 2] + bboxes1[:, 0]) / 2) #（x1max +x1min）/2
    # center_y1 = ((bboxes1[:, 3] + bboxes1[:, 1]) / 2)  #(y1max+y1min)/2
    # center_x2 = ((bboxes2[:, 2] + bboxes2[:, 0]) / 2)
    # center_y2 = ((bboxes2[:, 3] + bboxes2[:, 1]) / 2)

    center_x1 = (b1x2 + b1x1) / 2
    center_y1 = (b1y2 + b1y1) / 2
    center_x2 = (b2x2 + b2x1) / 2
    center_y2 = (b2y2 + b2y1) / 2

    # inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:]) #min((x1max,y1max ),(x2max,y2max)) ->返回较小一组
    # inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2]) ##max((x1min,y1min ),(x2min,y2min))->返回较大的一组
    # out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    # out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])
    cx = (b1x2.max(b2x2.t()) - b1x1.min(b2x1.t())).clamp(min=0)
    cy = (b1y2.max(b2y2.t()) - b1y1.min(b2y1.t())).clamp(min=0)
    # inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    # inter_area = inter[:, 0] * inter[:, 1]
    # inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    inter_diag = (center_x2.t() - center_x1) ** 2 + (center_y2.t() - center_y1) ** 2
    # outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    # outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    outer_diag = (cx ** 2) + (cy ** 2)
    # union = area1 + area2 - inter_area
    dious = inter_area / union - (inter_diag) / outer_diag
    dious = torch.clamp(dious, min=-1.0, max=1.0)
    # if exchange:
    #     dious = dious.T
    return dious

def bbox_overlaps_ciou_1(bboxes1, bboxes2):   #原CIoU代码
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return cious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = torch.zeros((cols, rows))
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:],bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2],bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1+area2-inter_area
    u = (inter_diag) / outer_diag
    iou = inter_area / union
    with torch.no_grad():
        arctan = torch.atan(w2 / h2) - torch.atan(w1 / h1)
        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
        S = 1 - iou
        alpha = v / (S + v)
        w_temp = 2 * w1
    ar = (8 / (math.pi ** 2)) * arctan * ((w1 - w_temp) * h1)
    cious = iou - (u + alpha * ar)
    cious = torch.clamp(cious,min=-1.0,max = 1.0)
    if exchange:
        cious = cious.T
    return cious


def bbox_overlaps_ciou(bboxes1, bboxes2):

    b1x1, b1y1 = (bboxes1[:, :2] - (bboxes1[:, 2:4] / 2)).split(1, 1)
    b1x2, b1y2 = (bboxes1[:, :2] + (bboxes1[:, 2:4] / 2)).split(1, 1)
    b2x1, b2y1 = (bboxes2[:, :2] - (bboxes2[:, 2:4] / 2)).split(1, 1)
    b2x2, b2y2 = (bboxes2[:, :2] + (bboxes2[:, 2:4] / 2)).split(1, 1)

    dx = (b1x2.min(b2x2.t()) - b1x1.max(b2x1.t())).clamp(min=0)
    dy = (b1y2.min(b2y2.t()) - b1y1.max(b2y1.t())).clamp(min=0)
    intersections = dx * dy

    areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)
    unions = (areas1 + areas2.t()) - intersections
    iou = intersections / unions

    # w1 = bboxes1[:, 2] - bboxes1[:, 0]
    # h1 = bboxes1[:, 3] - bboxes1[:, 1]
    # w2 = bboxes2[:, 2] - bboxes2[:, 0]
    # h2 = bboxes2[:, 3] - bboxes2[:, 1]

    # area1 = w1 * h1
    # area2 = w2 * h2
    w1 = (b1x2 - b1x1)
    h1 = (b1y2 - b1y1)
    w2 = (b2x2 - b2x1)
    h2 = (b2y2 - b2y1)

    # center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    # center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    # center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    # center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2
    center_x1 = (b1x2 + b1x1) / 2
    center_y1 = (b1y2 + b1y1) / 2
    center_x2 = (b2x2 + b2x1) / 2
    center_y2 = (b2y2 + b2y1) / 2
    # inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:])
    # inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2])
    # out_max_xy = torch.max(bboxes1[:, 2:],bboxes2[:, 2:])
    # out_min_xy = torch.min(bboxes1[:, :2],bboxes2[:, :2])
    cx = (b1x2.max(b2x2.t()) - b1x1.min(b2x1.t())).clamp(min=0)
    cy = (b1y2.max(b2y2.t()) - b1y1.min(b2y1.t())).clamp(min=0)


    # inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    # inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2.t() - center_x1)**2 + (center_y2.t() - center_y1)**2
    # outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    # outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    outer_diag = (cx ** 2) + (cy ** 2)
    # union = area1+area2-inter_area
    u = (inter_diag) / outer_diag
    # iou = inter_area / union
    with torch.no_grad():
        arctan = torch.atan(w2 / h2).t() - torch.atan(w1 / h1)
        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2).t() - torch.atan(w1 / h1)), 2)
        S = 1 - iou
        alpha = v / (S + v)
        w_temp = 2 * w1
    ar = (8 / (math.pi ** 2)) * arctan * ((w1 - w_temp) * h1)
    cious = iou - (u + alpha * ar)
    cious = torch.clamp(cious,min=-1.0,max = 1.0)
    # if exchange:
    #     cious = cious.T
    return cious