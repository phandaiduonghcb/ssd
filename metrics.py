import tensorflow as tf
import numpy as np
from datasets.face_dataset.face_dataset import CLASS_DICT
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def cls_eval(cls_preds, cls_labels): # 
    # Because the class prediction results are on the final dimension,
    # `argmax` needs to specify this 
    pred_result = tf.cast(tf.argmax(cls_preds, axis=-1),dtype=tf.float32)
    cls_labels = tf.cast(cls_labels, tf.float32)
    boolean =tf.cast((pred_result == cls_labels),dtype=tf.float32)
    result = tf.reduce_sum(boolean)
    return float(result)

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):

    bbox_masks = tf.cast(bbox_masks, bbox_labels.dtype)
    result = tf.math.abs((bbox_labels - bbox_preds) * bbox_masks)
    return float(tf.reduce_sum(result))