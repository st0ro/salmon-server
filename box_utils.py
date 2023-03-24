import collections
import itertools
import math
from typing import List
import tensorflow as tf
import numpy as np

SSDBoxSizes = collections.namedtuple('SSDBoxSizes', ['min', 'max'])

SSDSpec = collections.namedtuple('SSDSpec', ['feature_map_size', 'shrinkage', 'box_sizes', 'aspect_ratios'])


def generate_ssd_priors(specs: List[SSDSpec], image_size, clamp=True):
    """Generate SSD Prior Boxes.
    It returns the center, height and width of the priors. The values are relative to the image size
    Args:
        specs: SSDSpecs about the shapes of sizes of prior boxes. i.e.
            specs = [
                SSDSpec(38, 8, SSDBoxSizes(30, 60), [2]),
                SSDSpec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
                SSDSpec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
                SSDSpec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
                SSDSpec(3, 100, SSDBoxSizes(213, 264), [2]),
                SSDSpec(1, 300, SSDBoxSizes(264, 315), [2])
            ]
        image_size: image size.
        clamp: if true, clamp the values to make fall between [0.0, 1.0]
    Returns:
        priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
            are relative to the image size.
    """
    priors = []
    for spec in specs:
        scale = image_size / spec.shrinkage
        for j, i in itertools.product(range(spec.feature_map_size), repeat=2):
            x_center = (i + 0.5) / scale
            y_center = (j + 0.5) / scale

            # small sized square box
            size = spec.box_sizes.min
            h = w = size / image_size
            priors.append([
                x_center,
                y_center,
                w,
                h
            ])

            # big sized square box
            size = math.sqrt(spec.box_sizes.max * spec.box_sizes.min)
            h = w = size / image_size
            priors.append([
                x_center,
                y_center,
                w,
                h
            ])

            # change h/w ratio of the small sized box
            size = spec.box_sizes.min
            h = w = size / image_size
            for ratio in spec.aspect_ratios:
                ratio = math.sqrt(ratio)
                priors.append([
                    x_center,
                    y_center,
                    w * ratio,
                    h / ratio
                ])
                priors.append([
                    x_center,
                    y_center,
                    w / ratio,
                    h * ratio
                ])

    priors = tf.constant(priors)
    if clamp:
        priors = tf.clip_by_value(priors, 0.0, 1.0)
    return priors


def convert_locations_to_boxes(locations, priors, center_variance,
                               size_variance):
    """Convert regressional location results of SSD into boxes in the form of (center_x, center_y, h, w).
    The conversion:
        $$predicted\_center * center_variance = \frac {real\_center - prior\_center} {prior\_hw}$$
        $$exp(predicted\_hw * size_variance) = \frac {real\_hw} {prior\_hw}$$
    We do it in the inverse direction here.
    Args:
        locations (batch_size, num_priors, 4): the regression output of SSD. It will contain the outputs as well.
        priors (num_priors, 4) or (batch_size/1, num_priors, 4): prior boxes.
        center_variance: a float used to change the scale of center.
        size_variance: a float used to change of scale of size.
    Returns:
        boxes:  priors: [[center_x, center_y, h, w]]. All the values
            are relative to the image size.
    """
    # priors can have one dimension less.
    if len(priors.shape) + 1 == len(locations.shape):
        priors = tf.keras.backend.expand_dims(priors, 0)
    loc_slice = tf.slice(locations, [0, 0, 2], [-1, -1, -1])
    prior_slice = tf.slice(priors, [0, 0, 2], [-1, -1, -1])
    return tf.keras.layers.Concatenate(axis=len(locations.shape) - 1)([
        tf.slice(locations, [0, 0, 0], [-1, -1, 2]) * center_variance * prior_slice + tf.slice(priors, [0, 0, 0],
                                                                                               [-1, -1, 2]),
        tf.math.exp(loc_slice * size_variance) * prior_slice
    ])


def np_convert_locations_to_boxes(locations, priors, center_variance,
                                  size_variance):
    """Convert regressional location results of SSD into boxes in the form of (center_x, center_y, h, w).
    The conversion:
        $$predicted\_center * center_variance = \frac {real\_center - prior\_center} {prior\_hw}$$
        $$exp(predicted\_hw * size_variance) = \frac {real\_hw} {prior\_hw}$$
    We do it in the inverse direction here.
    Args:
        locations (batch_size, num_priors, 4): the regression output of SSD. It will contain the outputs as well.
        priors (num_priors, 4) or (batch_size/1, num_priors, 4): prior boxes.
        center_variance: a float used to change the scale of center.
        size_variance: a float used to change of scale of size.
    Returns:
        boxes:  priors: [[center_x, center_y, h, w]]. All the values
            are relative to the image size.
    """
    # priors can have one dimension less.
    if len(priors.shape) + 1 == len(locations.shape):
        priors = np.expand_dims(priors, 0)
    loc_slice = locations[..., 2:]
    prior_slice = priors[..., 2:]
    loc_1 = locations[..., :2]
    priors_1 = priors[..., :2]
    return np.concatenate(
        [loc_1 * center_variance * prior_slice + priors_1, np.exp(loc_slice * size_variance) * prior_slice],
        axis=len(locations.shape) - 1)


def convert_boxes_to_locations(center_form_boxes, center_form_priors, center_variance, size_variance):
    if len(center_form_priors.shape) + 1 == len(center_form_boxes.shape):
        center_form_priors = center_form_priors.unsqueeze(0)
    return tf.concat([
        (center_form_boxes[..., :2] - center_form_priors[..., :2]) / center_form_priors[..., 2:] / center_variance,
        tf.math.log(center_form_boxes[..., 2:] / center_form_priors[..., 2:]) / size_variance
    ], len(center_form_boxes.shape) - 1)


def area_of(left_top, right_bottom):
    """Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = tf.nn.relu(right_bottom - left_top)
    return tf.cast(hw[..., 0] * hw[..., 1], tf.float32)


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """

    overlap_left_top = tf.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = tf.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def hard_negative_mining(loss, labels, neg_pos_ratio):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.
    Args:
        loss (N, num_priors): the loss for each example.
        labels (N, num_priors): the labels.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """
    pos_mask = tf.math.greater(labels, 0)
    num_pos = tf.reduce_sum(tf.cast(pos_mask, dtype=tf.int32), axis=1, keepdims=True)
    num_neg = num_pos * neg_pos_ratio

    idxs = tf.where(pos_mask)
    loss = tf.tensor_scatter_nd_update(loss, idxs, tf.fill(tf.shape(idxs[:, 0]), -math.inf))
    indexes = tf.argsort(loss, axis=1, direction='DESCENDING')
    orders = tf.argsort(indexes, axis=1)
    neg_mask = tf.math.less(orders, num_neg)
    return tf.math.logical_or(pos_mask, neg_mask)


def center_form_to_corner_form(locations):
    if len(locations.shape) == 3:
        loc_slice_first = locations[:, :, :2]
        loc_slice_second = locations[:, :, 2:]
        return tf.concat([loc_slice_first - loc_slice_second / 2,
                          loc_slice_first + loc_slice_second / 2], len(locations.shape) - 1)

    return tf.concat([locations[..., :2] - locations[..., 2:] / 2,
                      locations[..., :2] + locations[..., 2:] / 2], len(locations.shape) - 1)


def np_center_form_to_corner_form(locations):
    if len(locations.shape) == 3:
        loc_slice_first = locations[:, :, :2]
        loc_slice_second = locations[:, :, 2:]
        return np.concatenate([loc_slice_first - loc_slice_second / 2, loc_slice_first + loc_slice_second / 2],
                              axis=len(locations.shape) - 1)

    return np.concatenate([locations[..., :2] - locations[..., 2:] / 2,
                           locations[..., :2] + locations[..., 2:] / 2], axis=len(locations.shape) - 1)


def corner_form_to_center_form(boxes):
    return tf.concat([
        (boxes[..., :2] + boxes[..., 2:]) / 2,
        boxes[..., 2:] - boxes[..., :2]
    ], len(boxes.shape) - 1)


def assign_priors(gt_boxes,
                  gt_labels,
                  corner_form_priors,
                  iou_threshold):
    """
    Assign ground truth boxes and targets to priors.
    :param gt_boxes: (num_targets, 4) ground truth boxes.
    :param gt_labels: (num_targets) labels of targets.
    :param corner_form_priors: (num_priors, 4) corner form priors
    :param iou_threshold: Minimum IOU threshold to consider a prior a match
    :return: boxes (num_priors, 4): real values for priors.
             labels (num_priors): labels for priors.
    """
    # size: num_priors x num_targets
    ious = iou_of(tf.expand_dims(gt_boxes, 0), tf.expand_dims(corner_form_priors, 1))
    # size: num_priors
    best_target_per_prior_index = tf.subtract(tf.shape(ious, out_type=tf.int32)[1] - 1,
                                              tf.argmax(tf.reverse(ious, axis=[1]), 1, output_type=tf.int32))
    best_target_per_prior = tf.math.reduce_max(ious, 1)
    # size: num_targets
    best_prior_per_target_index = tf.subtract(tf.shape(ious, out_type=tf.int32)[0] - 1,
                                              tf.argmax(tf.reverse(ious, axis=[0]), 0, output_type=tf.int32))
    del ious  # Free up ious memory
    best_target_per_prior_index = tf.tensor_scatter_nd_update(
        best_target_per_prior_index,
        tf.expand_dims(best_prior_per_target_index, 1),
        tf.range(tf.shape(best_prior_per_target_index)[0])
    )

    # 2.0 is used to make sure every target has a prior assigned
    best_target_per_prior = tf.tensor_scatter_nd_update(
        best_target_per_prior,
        tf.expand_dims(best_prior_per_target_index, 1),
        tf.fill(tf.shape(best_prior_per_target_index), 2.0)
    )

    # size: num_priors
    labels = tf.gather(gt_labels, best_target_per_prior_index)

    labels = tf.multiply(labels, tf.cast(tf.greater_equal(best_target_per_prior, iou_threshold), labels.dtype))
    boxes = tf.gather(gt_boxes, best_target_per_prior_index)
    return boxes, tf.expand_dims(tf.cast(labels, tf.float32), 1)


# def post_process(scores,
#                  boxes,
#                  use_tf_nms=True,
#                  coord_format='x_first',
#                  top_k=-1,
#                  prob_threshold=0.5,
#                  iou_threshold=0.5,
#                  candidate_size=200):
#     boxes = boxes[0]
#     scores = scores[0]
#     picked_box_probs = []
#     picked_labels = []
#     if use_tf_nms:
#         boxes = tf.stack([
#             boxes[:, 1],
#             boxes[:, 0],
#             boxes[:, 3],
#             boxes[:, 2]
#         ], axis=1)
#
#     for class_index in range(1, scores.shape[1]):
#         probs = scores[:, class_index]
#         mask = probs > prob_threshold
#         probs = probs[mask]
#         if probs.shape[0] == 0:
#             continue
#         subset_boxes = tf.boolean_mask(boxes, mask, axis=0)
#         if use_tf_nms:
#             selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
#                 subset_boxes,
#                 probs,
#                 100,
#                 iou_threshold=iou_threshold,
#                 score_threshold=prob_threshold
#             )
#             selected_boxes = tf.gather(subset_boxes, selected_indices)
#             box_probs = tf.concat([selected_boxes, tf.expand_dims(selected_scores, axis=1)],
#                                   axis=1)
#         else:
#             box_probs = tf.concat([subset_boxes, tf.reshape(probs, (-1, 1))], axis=1)
#             box_probs = nms(box_probs,
#                             'hard',
#                             score_threshold=prob_threshold,
#                             iou_threshold=iou_threshold,
#                             top_k=top_k,
#                             candidate_size=candidate_size)
#         picked_box_probs.append(box_probs)
#         picked_labels.extend([class_index] * box_probs.shape[0])
#
#     if not picked_box_probs:
#         return tf.constant([], dtype=tf.float32), tf.constant([], dtype=tf.float32), tf.constant([],
#                                                                                                  dtype=tf.float32)
#     picked_box_probs_temp = tf.concat(picked_box_probs, axis=0)
#     if coord_format == 'x_first':
#         if use_tf_nms:
#             coords = [1, 0, 3, 2]
#         else:
#             coords = [0, 1, 2, 3]
#     else:  # y_first
#         if use_tf_nms:
#             coords = [0, 1, 2, 3]
#         else:
#             coords = [1, 0, 3, 2]
#
#     picked_box_probs = [picked_box_probs_temp[:, coords[0]], picked_box_probs_temp[:, coords[1]],
#                         picked_box_probs_temp[:, coords[2]], picked_box_probs_temp[:, coords[3]],
#                         picked_box_probs_temp[:, 4]]
#     picked_box_probs = tf.stack(picked_box_probs, axis=1)
#
#     return picked_box_probs[:, :4], tf.constant(picked_labels), picked_box_probs[:, 4]


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = tf.argsort(scores, direction='DESCENDING')
    indexes = indexes[:candidate_size]
    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current.numpy())
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = tf.gather(boxes, indexes)
        iou = iou_of(
            rest_boxes,
            tf.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou >= iou_threshold]

    return tf.gather(box_scores, picked)


def nms(box_scores, nms_method=None, score_threshold=None, iou_threshold=None,
        sigma=0.5, top_k=-1, candidate_size=200):
    if nms_method == "soft":
        return soft_nms(box_scores, score_threshold, sigma, top_k)
    else:
        return hard_nms(box_scores, iou_threshold, top_k, candidate_size=candidate_size)


def soft_nms(box_scores, score_threshold, sigma=0.5, top_k=-1):
    """Soft NMS implementation.
    References:
        https://arxiv.org/abs/1704.04503
        https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/cython_nms.pyx
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        score_threshold: boxes with scores less than value are not considered.
        sigma: the parameter in score re-computation.
            scores[i] = scores[i] * exp(-(iou_i)^2 / simga)
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
         picked_box_scores (K, 5): results of NMS.
    """
    picked_box_scores = []
    while box_scores.size(0) > 0:
        max_score_index = tf.argmax(box_scores[:, 4])
        cur_box_prob = tf.constant(box_scores[max_score_index, :])
        picked_box_scores.append(cur_box_prob)
        if len(picked_box_scores) == top_k > 0 or box_scores.size(0) == 1:
            break
        cur_box = cur_box_prob[:-1]
        box_scores[max_score_index, :] = box_scores[-1, :]
        box_scores = box_scores[:-1, :]
        ious = iou_of(cur_box.unsqueeze(0), box_scores[:, :-1])
        box_scores[:, -1] = box_scores[:, -1] * tf.exp(-(ious * ious) / sigma)
        box_scores = box_scores[box_scores[:, -1] > score_threshold, :]
    if len(picked_box_scores) > 0:
        return tf.stack(picked_box_scores)
    else:
        return tf.constant([])