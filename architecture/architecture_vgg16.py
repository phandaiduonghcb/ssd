import tensorflow as tf
from anchor import multibox_prior
import math
def generate_sizes(n=5,start=0.2,end=1.05):
    sizes = []
    t = (end - start) / n
    for i in range(n):
        if i==0:
            s_0 = start
            # s_1 = start + tf.sqrt(s_0*(s_0 + t))
            # sizes = [[s_0, s_1]]
        else:
            s_0 = sizes[-1][0] + t
        s_1 = math.sqrt(s_0*(s_0 + t))
        sizes.append([s_0, s_1])
    return sizes

## HYPERPARAMS
SIZES = generate_sizes(6,0.2,1.10)
RATIOS = [[1, 2, 0.5]] * 6
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

def vgg_block(num_convs, num_channels):
    blk = tf.keras.models.Sequential()
    for _ in range(num_convs):
        blk.add(
            tf.keras.layers.Conv2D(num_channels, kernel_size=3,
                                   padding='same', activation='relu'))
    blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2,padding="same"))
    return blk
def conv4_3():
    return tf.keras.models.Sequential([
        vgg_block(2,64),
        vgg_block(2,128),
        vgg_block(3,256),
        tf.keras.layers.Conv2D(512, kernel_size=3,
                                   padding='same', activation='relu'),
        tf.keras.layers.Conv2D(512, kernel_size=3,
                                   padding='same', activation='relu'),
        tf.keras.layers.Conv2D(512, kernel_size=3,
                                   padding='same', activation='relu')
    ])

def fc_7():
    return tf.keras.models.Sequential([
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(512, kernel_size=3,
                                   padding='same', activation='relu'),
        tf.keras.layers.Conv2D(512, kernel_size=3,
                                   padding='same', activation='relu'),
        tf.keras.layers.Conv2D(512, kernel_size=3,
                                   padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(3,1,padding='same'),
        tf.keras.layers.Conv2D(1024,3,padding="same"),
        tf.keras.layers.Conv2D(1024,1,padding="same"),
    ])

def conv8_2():
    return tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(256,1,padding="same"),
    tf.keras.layers.Conv2D(512,3,padding="same", strides=2),
    ])

def conv9_2():
    return tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(128,1,padding="same"),
    tf.keras.layers.Conv2D(256,3,padding="same", strides=2),
    ])
def conv10_2():
    return conv9_2()

def conv11_2():
    return tf.keras.layers.GlobalAveragePooling2D(keepdims=True)

def get_blk(i): # 
    if i==0:
        blk = conv4_3()
    elif i== 1: blk =fc_7()
    elif i==2: blk=conv8_2()
    elif i==3: blk=conv9_2()
    elif i==4: blk=conv10_2()
    elif i==5: blk=conv11_2()
    return blk

def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)

class SSD(tf.keras.models.Model):
    def __init__(self, num_classes, **kwargs):
        super(SSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        
        for i in range(6): # HYPERPARAMS
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(NUM_ANCHOR, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(NUM_ANCHOR))

    def call(self, X):
        anchors, cls_preds, bbox_preds = [None]*6, [None]*6, [None]*6
        for i in range(6): # HYPERPARAMS
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), SIZES[i], RATIOS[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}')
            )
        anchors = tf.concat(anchors, axis=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = tf.reshape(cls_preds, (-1,(38**2 + 19**2 + 10**2 + 5**2 + 3**2 + 1)*NUM_ANCHOR,self.num_classes + 1))
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds