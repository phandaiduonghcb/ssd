import tensorflow as tf
from anchor import offset_inverse
from focal_loss import sparse_categorical_focal_loss
ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
l1_loss = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)

def calc_loss(anchors, cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks, neg_pos_ratio=0, width=300, height=300):

    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    n_neg_min=0
    alpha = 1

    # classification_loss = ce_loss(cls_labels, cls_preds)
    classification_loss = sparse_categorical_focal_loss(cls_labels, cls_preds, gamma=2, from_logits=True)
    positive_indices = tf.where(cls_labels > 0)
    n_positive = tf.shape(positive_indices)[0]

    pos_class_loss = tf.reduce_sum(tf.cast(cls_labels > 0, tf.float32) * classification_loss, axis=1)
    neg_class_loss_all = classification_loss * tf.cast(cls_labels == 0, tf.float32)
    n_neg_losses = tf.reduce_sum(tf.cast(neg_class_loss_all == 0, tf.float32))
    n_negative_keep = tf.minimum(tf.maximum(neg_pos_ratio * n_positive,n_neg_min), tf.cast(n_neg_losses, tf.int32))

    def f1():
        return tf.zeros((batch_size,))
    def f2():
        neg_class_loss_all_1D = tf.reshape(neg_class_loss_all, [-1])
        values, indices = tf.nn.top_k(neg_class_loss_all_1D,
                                          k=n_negative_keep,
                                          sorted=False)

        negatives_keep = tf.scatter_nd(indices=tf.expand_dims(indices, axis=1),
                                           updates=tf.ones_like(indices, dtype=tf.int32),
                                           shape=tf.shape(neg_class_loss_all_1D)) 

        negatives_keep = tf.cast(tf.reshape(negatives_keep, [batch_size, -1]), tf.float32)
        neg_class_loss = tf.reduce_sum(classification_loss * negatives_keep, axis=-1) # Tensor of shape (batch_size,)
        return neg_class_loss

    neg_class_loss = tf.cond(tf.equal(n_neg_losses, tf.constant(0.)), f1, f2)
    class_loss = pos_class_loss + neg_class_loss
    #
    bbox_preds = tf.reshape(bbox_preds, (batch_size,-1,4))
    bbox_labels =  tf.reshape(bbox_labels, (batch_size,-1,4))
    bbox_masks = tf.reshape(bbox_masks, (batch_size, -1,4))
    
    corner_bbox_preds = []
    corner_bbox_labels = []
    anchors = tf.squeeze(anchors)
    for i in range(batch_size):
        bbox_pred = offset_inverse(anchors, bbox_preds[i])
        bbox_label = offset_inverse(anchors, bbox_labels[i])
        corner_bbox_preds.append(bbox_pred)
        corner_bbox_labels.append(bbox_label)

    corner_bbox_labels = tf.convert_to_tensor(corner_bbox_labels, tf.float32)
    corner_bbox_preds = tf.convert_to_tensor(corner_bbox_preds, tf.float32)

    t = tf.convert_to_tensor([width, height, width, height], tf.float32)
    localization_loss = smooth_L1_loss(corner_bbox_labels * t , corner_bbox_preds*t)
    loc_loss = tf.reduce_sum(localization_loss*bbox_masks[:,:,0], axis=1)
    total_loss = (class_loss + alpha * loc_loss) / tf.maximum(tf.constant(1), n_positive)
    # total_loss = total_loss * tf.cast(batch_size, tf.float32)
    return total_loss

def smooth_L1_loss(y_true, y_pred):
    '''
    Compute smooth L1 loss, see references.
    Arguments:
        y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
            In this context, the expected tensor has shape `(batch_size, #boxes, 4)` and
            contains the ground truth bounding box coordinates, where the last dimension
            contains `(xmin, xmax, ymin, ymax)`.
        y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
            the predicted data, in this context the predicted bounding box coordinates.
    Returns:
        The smooth L1 loss, a nD-1 Tensorflow tensor. In this context a 2D tensor
        of shape (batch, n_boxes_total).
    References:
        https://arxiv.org/abs/1504.08083
    '''
    absolute_loss = tf.abs(y_true - y_pred)
    square_loss = 0.5 * (y_true - y_pred)**2
    l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
    return tf.reduce_sum(l1_loss, axis=-1)

if __name__ == "__main__":
    cls_preds = tf.cast(tf.random.uniform((2,2,2)), tf.float32)
    cls_labels = tf.cast(tf.random.uniform((2,2), minval=0, maxval=2, dtype=tf.int32), tf.float32)
    bbox_preds = tf.random.uniform((2,2*4))
    bbox_labels = tf.random.uniform(bbox_preds.shape)
    bbox_masks = tf.ones(bbox_preds.shape)
    # class_loss(cls_labels, cls_preds)
    print(calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks,3).shape)