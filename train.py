import tensorflow   as tf
from losses import calc_loss
from anchor import multibox_target
from metrics import cls_eval, bbox_eval
from architecture.architecture_vgg16 import SSD
from utils import LogWriter
# from datasets.face_dataset.face_dataset import CLASS_DICT, NUM_TRAIN_EXAMPLES
# import datasets.face_dataset
from datasets.augmented_face_dataset.augmented_face_dataset import CLASS_DICT, NUM_TRAIN_EXAMPLES
import datasets.augmented_face_dataset
import tensorflow_datasets as tfds
from batch import BatchDatasetForOD
import os
import shutil
import glob
import re
import numpy as np
import sys

@tf.function(reduce_retracing=True)
def training_step(net, X, Y):
    with tf.GradientTape() as tape:
        anchors, cls_preds, bbox_preds = net(X)
        bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
        c = tf.nn.softmax(cls_preds[0])
        t = tf.cast(c[:,1] >  0.5, tf.float32)
        k = tf.cast(c[:,1] <=  0.5, tf.float32)
        tf.print(tf.reduce_sum(t),tf.reduce_sum(k), sys.stderr)
        l = calc_loss(anchors, cls_preds, cls_labels, bbox_preds, bbox_labels,
                    bbox_masks, positive_negative_ratio, image_size[0], image_size[1])
        l_mean = tf.reduce_mean(l)
    grads = tape.gradient(l_mean, net.trainable_weights)
    optimizer.apply_gradients(zip(grads, net.trainable_weights))
    return l_mean

@tf.function(reduce_retracing=True)
def test_step(net, X, Y):
    anchors, cls_preds, bbox_preds = net(X, training=False)
    bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
    l = calc_loss(anchors, cls_preds, cls_labels, bbox_preds, bbox_labels,
                    bbox_masks, positive_negative_ratio, image_size[0], image_size[1])
    l_mean = tf.reduce_mean(l)
    acc_err = 1-(cls_eval(cls_preds, cls_labels)/float(tf.size(cls_labels)))
    mae = (bbox_eval(bbox_preds, bbox_labels, bbox_masks))/float(tf.size(bbox_labels))
    return l_mean, acc_err , mae 

num_epoch = 100
start_epoch = -1
save_model_interval = 5
model_dir = './trained_models/ssd_vgg16_fl_hnm_aug'
model_path = None #'/dl/ssd/trained_models/tiny_ssd/9_checkpoint.index' #'/dl/ssd/9_checkpoint'
BATCH_SIZE = 16
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=1e-2,
#     decay_steps=10000,
#     decay_rate=0.9)
# optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
positive_negative_ratio=5
optimizer = tf.keras.optimizers.SGD(learning_rate=0.005, decay=5e-4)
image_size = (300,300)
log_path = "./log/ssd_vgg16_fl_hnm_aug"
if not os.path.exists(log_path):
    os.makedirs(log_path)
logger = LogWriter(log_path = log_path)
strategy = tf.distribute.MirroredStrategy()
dataset = tfds.load('augmented_face_dataset')
min_val_loss = float('inf')


with strategy.scope():
    net = SSD(num_classes=len(CLASS_DICT.keys()))

if model_path is not None:
    net = tf.keras.models.load_model(model_path)
    start_epoch = int(model_path.split('/')[-1].split('_')[0])

for epoch in range(start_epoch+1,num_epoch):
    print(f'--- Epoch {epoch} ---')
    print(optimizer._decayed_lr(tf.float32))
    train_dataset = dataset['train'].shuffle(NUM_TRAIN_EXAMPLES)
    valid_dataset = dataset['valid']

    batched_train_dataset = BatchDatasetForOD(train_dataset, BATCH_SIZE, image_size)
    batched_valid_dataset = BatchDatasetForOD(valid_dataset, BATCH_SIZE,image_size)

    train_total_loss = 0
    for X,Y in batched_train_dataset:
      loss = training_step(net, X, Y)
      train_total_loss += loss
    
    total_acc_err = 0
    total_box_mae = 0
    val_total_loss = 0
    
    for X,Y in batched_valid_dataset:
      loss, acc_err, box_mae = test_step(net, X, Y)
      val_total_loss += loss
      total_acc_err += acc_err
      total_box_mae += box_mae

    print('train', 'total_loss', train_total_loss)
    print('val', 'total_loss', val_total_loss)
    print('val', 'box_mae', total_box_mae)
    print('val', 'acc_err', total_acc_err)

    if (epoch+1)%save_model_interval == 0:
        print(val_total_loss, min_val_loss)
        if val_total_loss < min_val_loss:
            print("Smallest val loss!!!")
            dst = os.path.join(model_dir, 'best_model')
            net.save(dst)
            min_val_loss = val_total_loss
        print('...Saving model...')
        dst = os.path.join(model_dir, f'{epoch}_model')
        net.save(dst)
        regex = re.compile(f"^{epoch - save_model_interval}_model")
        for file in glob.glob(os.path.join(model_dir, '*')):
            if regex.match(file.split('/')[-1]):
                shutil.rmtree(file)

    logger.add_a_point('train', 'total_loss', train_total_loss,epoch)
    logger.add_a_point('val', 'box_mae', total_box_mae,epoch)
    logger.add_a_point('val', 'acc_err', total_acc_err, epoch)
    logger.add_a_point('val', 'total_loss', val_total_loss, epoch)
