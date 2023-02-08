import tensorflow as tf
from anchor import multibox_prior

# HYPERPARAMS

SIZES = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
RATIOS = [[1, 2, 0.5]] * 5
NUM_ANCHOR = len(SIZES[0]) + len(RATIOS[0]) - 1

def cls_predictor(num_anchors, num_classes):
    return tf.keras.layers.Conv2D(num_anchors*(num_classes+1), kernel_size=(3,3), padding='same')

def bbox_predictor(num_anchors):
    return tf.keras.layers.Conv2D(num_anchors * 4, kernel_size=(3,3), padding='same')

def flatten_pred(pred):
    flat = tf.keras.layers.Flatten()
    return flat(pred)

def concat_preds(preds):
    return tf.concat([flatten_pred(pred) for pred in preds], axis =1)

def down_sample_blk(out_channels):
    blk = []
    for _ in range(2):
        blk.append(tf.keras.layers.Conv2D(out_channels,
                             kernel_size=3, padding='same'))
        blk.append(tf.keras.layers.BatchNormalization())
        blk.append(tf.keras.layers.ReLU())
    blk.append(tf.keras.layers.MaxPool2D(pool_size=2))
    return tf.keras.models.Sequential(blk)

def base_net():
    blk = []
    num_filters = [16, 32, 64] # HYPERPARAMS
    for f in num_filters:
        blk.append(down_sample_blk(f))
    return tf.keras.models.Sequential(blk)

def get_blk(i): # HYPERPARAMS
    if i==0:
        blk=base_net()
    elif i==1:
        blk = down_sample_blk(128)
    elif i==4:
        blk = tf.keras.layers.GlobalAveragePooling2D()
    else:
        blk = down_sample_blk(128)
    return blk

def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    if len(Y.shape) <4:
        Y =tf.expand_dims(Y, axis=1)
        Y =tf.expand_dims(Y, axis=1)
    anchors = multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)

class SSD(tf.keras.models.Model):
    def __init__(self, num_classes, **kwargs):
        super(SSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        
        for i in range(5): # HYPERPARAMS
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(NUM_ANCHOR, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(NUM_ANCHOR))

    def call(self, X):
        anchors, cls_preds, bbox_preds = [None]*5, [None]*5, [None]*5
        for i in range(5): # HYPERPARAMS
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), SIZES[i], RATIOS[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}')
            )
        anchors = tf.concat(anchors, axis=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = tf.reshape(cls_preds, (-1,(37**2 + 18**2 + 9**2 + 4**2 + 1)*NUM_ANCHOR,self.num_classes + 1))
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds