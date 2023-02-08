import tensorflow as tf
from utils import box_iou, offset_boxes, offset_inverse, nms
import time

def multibox_prior(data, sizes, ratios):
    in_height, in_width = data.shape[1:3]
    num_sizes, num_ratios = len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = tf.convert_to_tensor(sizes)
    ratio_tensor = tf.convert_to_tensor(ratios)
    # Offsets are required to move the anchor to the center of a pixel. Since
    # a pixel has height=1 and width=1, we choose to offset our centers by 0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Scaled steps in y axis
    steps_w = 1.0 / in_width  # Scaled steps in x axis

    # Generate all center points for the anchor boxes
    center_h = (tf.range(in_height, dtype=tf.float32) + offset_h) * steps_h
    center_w = (tf.range(in_width, dtype=tf.float32) + offset_w) * steps_w
    shift_y, shift_x = tf.meshgrid(center_h, center_w, indexing = 'ij')
    shift_y, shift_x = tf.reshape(shift_y,(-1,)), tf.reshape(shift_x,(-1,))
    # Generate `boxes_per_pixel` number of heights and widths that are later
    # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
    w = tf.concat((size_tensor * tf.sqrt(ratio_tensor[0]),
                   sizes[0] * tf.sqrt(ratio_tensor[1:])), axis=0)\
                   * in_height / in_width  # Handle rectangular inputs
    h = tf.concat((size_tensor / tf.sqrt(ratio_tensor[0]),
                   sizes[0] / tf.sqrt(ratio_tensor[1:])),axis=0)

    # # Divide by 2 to get half height and half width
    anchor_manipulations = tf.tile(tf.transpose(tf.stack((-w, -h, w, h))), (in_height * in_width,1)) / 2
    # anchor_manipulations = tf.stack((-w, -h, w, h)).T.repeat(
    #                                     in_height * in_width, 1) / 2
    # # Each center point will have `boxes_per_pixel` number of anchor boxes, so
    # # generate a grid of all anchor box centers with `boxes_per_pixel` repeats
    out_grid = tf.repeat(tf.stack([shift_x, shift_y, shift_x, shift_y], axis=1), boxes_per_pixel, axis=0)
    output = out_grid + anchor_manipulations
    return tf.expand_dims(output,0)
    # return output.unsqueeze(0)

def assign_anchor_to_bbox(ground_truth, anchors, iou_threshold=0.5):
    """Assign closest ground-truth bounding boxes to anchor boxes."""
    num_anchors, num_gt_boxes = tf.shape(anchors)[0], tf.shape(ground_truth)[0]
    # Element x_ij in the i-th row and j-th column is the IoU of the anchor
    # box i and the ground-truth bounding box j
    jaccard = box_iou(anchors, ground_truth)
    # Initialize the tensor to hold the assigned ground-truth bounding box for
    # each anchor   
    anchors_bbox_map = tf.fill((num_anchors,), -1)
    # Assign ground-truth bounding boxes according to the threshold
    max_ious = tf.reduce_max(jaccard, axis=1)
    indices = tf.argmax(jaccard, axis=1)
    anc_i = tf.reshape(tf.where(max_ious >= iou_threshold),(-1,1))
    box_j = indices[max_ious >= iou_threshold]
    box_j = tf.cast(box_j, anchors_bbox_map.dtype)
    anchors_bbox_map = tf.tensor_scatter_nd_update(anchors_bbox_map, anc_i, box_j)

    col_discard = tf.fill((1,num_anchors), -1)
    row_discard = tf.fill((1,num_gt_boxes,), -1)
    col_discard = tf.cast(col_discard, jaccard.dtype)
    row_discard = tf.cast(row_discard, jaccard.dtype)

    for _ in range(num_gt_boxes):
        max_idx = tf.argmax(tf.reshape(jaccard,(-1,)))  # Find the largest IoU
        max_idx = tf.cast(max_idx, tf.int32)
        box_idx = (max_idx % num_gt_boxes)
        anc_idx = max_idx // num_gt_boxes
        anchors_bbox_map = tf.tensor_scatter_nd_update(anchors_bbox_map, tf.reshape(anc_idx,(-1,1)), tf.reshape(tf.cast(box_idx, anchors_bbox_map.dtype),(-1,)))

        first_idxs = tf.reshape(tf.range(num_anchors, dtype=tf.int32),(-1,1))
        second_idxs = tf.fill([num_anchors,1], box_idx)
        second_idxs = tf.cast(second_idxs, first_idxs.dtype)
        col_idxs = tf.concat((first_idxs,second_idxs), axis=1)
        jaccard = tf.tensor_scatter_nd_update(jaccard, tf.expand_dims(col_idxs, 0), col_discard)
        
        first_idxs = tf.fill([num_gt_boxes,1], anc_idx)
        second_idxs = tf.reshape(tf.range(num_gt_boxes, dtype=first_idxs.dtype),(-1,1))
        row_idxs = tf.concat((first_idxs,second_idxs), axis=1)
        jaccard = tf.tensor_scatter_nd_update(jaccard, tf.expand_dims(row_idxs, 0), row_discard)
    return anchors_bbox_map

def multibox_target(anchors, labels):
    """Label anchor boxes using ground-truth bounding boxes."""
    batch_size, anchors = len(labels), tf.squeeze(anchors,0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    num_anchors = tf.shape(anchors)[0]
    for i in range(batch_size):
        label = labels[i]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors)
        bbox_mask = tf.repeat(tf.expand_dims(tf.cast((anchors_bbox_map >= 0),dtype=tf.float32), axis=-1),repeats=4,axis=1)
        # Initialize class labels and assigned bounding box coordinates with
        # zeros
        class_labels = tf.zeros(num_anchors, dtype=tf.float32)
        assigned_bb = tf.zeros((num_anchors, 4), dtype=tf.float32)
        # Label classes of anchor boxes using their assigned ground-truth
        # bounding boxes. If an anchor box is not assigned any, we label its
        # class as background (the value remains zero)
        indices_true = tf.where(anchors_bbox_map >= 0)
        bb_idx = tf.gather(anchors_bbox_map, (indices_true))
        
        chosen_label = tf.gather(label,tf.squeeze(bb_idx))

        if len(tf.shape(chosen_label)) == 1:
            class_labels_updates = tf.convert_to_tensor([int(chosen_label[0]) + 1], class_labels.dtype)
            assigned_bb_updates = tf.cast(tf.expand_dims(chosen_label[1:], axis=0), assigned_bb.dtype)
        else:
            class_labels_updates = tf.cast(chosen_label[:,0] +1, class_labels.dtype)
            assigned_bb_updates = tf.cast(chosen_label[:,1:], assigned_bb.dtype)
            
        class_labels = tf.tensor_scatter_nd_update(class_labels, indices_true , class_labels_updates)
        assigned_bb = tf.tensor_scatter_nd_update(assigned_bb, indices_true, assigned_bb_updates)
        
        # Offset transformation
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(tf.reshape(offset,(-1,)))
        batch_mask.append(tf.reshape(bbox_mask,(-1,)))
        batch_class_labels.append(class_labels)
    bbox_offset = tf.stack(batch_offset)
    bbox_mask = tf.stack(batch_mask)
    class_labels = tf.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)

def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """Predict bounding boxes using non-maximum suppression."""
    batch_size = cls_probs.shape[0]
    anchors = tf.squeeze(anchors)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], tf.reshape(offset_preds[i],((-1, 4)))
        conf= tf.reduce_max(cls_prob[1:], 0)
        class_id = tf.argmax(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = tf.image.non_max_suppression(
            tf.stack((predicted_bb[:,1], predicted_bb[:,0], predicted_bb[:,3], predicted_bb[:,2]), axis=1), conf,max_output_size=300,iou_threshold=nms_threshold)
        # keep = nms(predicted_bb, conf, nms_threshold)
        # Find all non-`keep` indices and set the class to background
        all_idx = tf.range(num_anchors, dtype=tf.int32)
        keep = tf.cast(keep,all_idx.dtype)
        combined = tf.concat((keep, all_idx),axis=0)
        uniques, idxs, counts = tf.unique_with_counts(combined)
        non_keep = uniques[counts == 1]
        all_id_sorted = tf.concat((keep, non_keep), axis=0)
        # class_id[non_keep] = -1
        class_id_updates = tf.cast(tf.fill((tf.size(non_keep),),-1), class_id.dtype)
        class_id = tf.tensor_scatter_nd_update(class_id, tf.expand_dims(non_keep, axis=1), class_id_updates)
        class_id = tf.gather(class_id, all_id_sorted)
        conf, predicted_bb = tf.gather(conf,all_id_sorted), tf.gather(predicted_bb ,all_id_sorted)
        # Here `pos_threshold` is a threshold for positive (non-background)
        # predictions
        below_min_idx = (conf < pos_threshold)
        class_id = tf.where(below_min_idx, tf.cast(tf.fill((tf.shape(conf)),-1),class_id.dtype), class_id)
        
        temp = tf.where(below_min_idx, tf.cast(tf.fill((tf.shape(conf)),-1),conf.dtype), 1 - conf)
        conf = tf.where(below_min_idx, temp, conf)
        pred_info = tf.concat((tf.cast(tf.expand_dims(class_id,1),tf.float32), tf.cast(tf.expand_dims(conf,1),tf.float32), tf.cast(predicted_bb,tf.float32)), axis =1)
        out.append(pred_info)
    return tf.stack(out)