import os
import time
import tensorflow as tf
from d2l import tensorflow as d2l

def show_bboxes(axes, bboxes, labels=None, colors=None):
    """Show bounding boxes."""

    def make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = make_list(labels)
    colors = make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(bbox.numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))
                      
def box_iou(boxes1, boxes2):
    """Compute pairwise IoU across two lists of anchor or bounding boxes."""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # Shape of `boxes1`, `boxes2`, `areas1`, `areas2`: (no. of boxes1, 4),
    # (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`: (no. of
    # boxes1, no. of boxes2, 2)
    inter_upperlefts = tf.maximum(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = tf.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts)
    inters = tf.clip_by_value(inters,clip_value_min=0,clip_value_max=inters.dtype.max)
    # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas

def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """Transform for anchor box offsets."""
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * tf.math.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = tf.concat([offset_xy, offset_wh], axis=1)
    return offset

def offset_inverse(anchors, offset_preds):
    """Predict bounding boxes based on anchor boxes with predicted offsets."""
    anc = d2l.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = tf.math.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = tf.concat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox

def nms(boxes, scores, iou_threshold):
    """Sort confidence scores of predicted bounding boxes."""
    B = tf.argsort(scores,axis=-1, direction= 'DESCENDING')
    keep = []  # Indices of predicted bounding boxes that will be kept
    while tf.size(B) > 0:
        i = B[0]
        keep.append(i)
        if tf.size(B) == 1: break
        iou = box_iou(tf.reshape(boxes[i, :], (-1, 4)),
                      tf.reshape(tf.gather(boxes,B[1:]),(-1,4)))
        iou = tf.reshape(iou, -1)
        inds = tf.reshape(tf.where(iou <= iou_threshold),(-1,))
        B = tf.gather(B,inds+1)
    return tf.convert_to_tensor(keep)

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().

    Defined in :numref:`sec_use_gpu`"""
    if len(tf.config.experimental.list_physical_devices('GPU')) >= i + 1:
        return tf.device(f'/GPU:{i}')
    return tf.device('/CPU:0')


class LogWriter():
    def __init__(self,log_path, y_names=['accuracy, loss'], folders=['train','val'],):
        self.log_path = log_path
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        self.folders = folders
        self.times = []

        self.summary_writers = {}
        self.y_values = {}
        for y in y_names:
            self.y_values[y] = []
        for folder in folders:
            self.summary_writers[folder]=(tf.summary.create_file_writer(os.path.join(log_path, folder)))
        
    def add_a_point(self, folder, y_name, y_value, x_value):
        with self.summary_writers[folder].as_default():
            tf.summary.scalar(y_name, y_value, x_value)

    def start_timing(self):
        self.start_time = time.time()
    
    def end_timing(self):
        self.times.append(time.time() - self.start_time)