import collections
import numpy as np
import tensorflow as tf

SSDBoxSizes = collections.namedtuple('SSDBoxSizes', ['height', 'width'])

SSDSpec = collections.namedtuple('SSDSpec', ['feature_map_size', 'scales', 'aspect_ratios'])


def generate_ssd_specs(box_specs, feature_map_sizes):
    specs = []
    scales = []
    aspect_ratios = []
    for layer in box_specs:
        temp_scales = []
        temp_aspect_ratios = []
        for item in layer:
            temp_scales.append(item[0])
            temp_aspect_ratios.append(item[1])
        scales.append(temp_scales)
        aspect_ratios.append(temp_aspect_ratios)

    for feature_size, scale, aspect_ratio in zip(feature_map_sizes, scales, aspect_ratios):
        specs.append(SSDSpec(feature_size, scale, aspect_ratio))
    return specs


def generate_ssd_priors(box_specs, feature_map_sizes, clamp=False):
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
    specs = generate_ssd_specs(box_specs, feature_map_sizes)
    priors = []
    for spec in specs:
        ratio_sqrts = tf.sqrt(spec.aspect_ratios)
        heights = spec.scales / ratio_sqrts
        widths = spec.scales * ratio_sqrts
        stride = 1.0 / spec.feature_map_size
        offset = 0.5 * stride

        # Get a grid of box centers
        y_centers = tf.cast(tf.range(spec.feature_map_size), dtype=tf.float32)
        y_centers = y_centers * stride + offset
        x_centers = tf.cast(tf.range(spec.feature_map_size), dtype=tf.float32)
        x_centers = x_centers * stride + offset
        x_centers, y_centers = tf.meshgrid(x_centers, y_centers)

        widths_grid, x_centers_grid = tf.meshgrid(widths, x_centers)
        heights_grid, y_centers_grid = tf.meshgrid(heights, y_centers)
        bbox_centers = tf.stack([y_centers_grid, x_centers_grid], axis=2)
        bbox_sizes = tf.stack([heights_grid, widths_grid], axis=2)
        bbox_centers = tf.reshape(bbox_centers, [-1, 2])
        bbox_sizes = tf.reshape(bbox_sizes, [-1, 2])
        priors.append(tf.concat([bbox_centers, bbox_sizes], 1))
    priors = tf.concat(priors, 0)
    if clamp:
        priors = tf.clip_by_value(priors, 0.0, 1.0)
    return priors


image_size = 320
image_mean = np.array([127.5, 127.5, 127.5])  # RGB layout
image_std = 127.5

center_variance = 0.1
size_variance = 0.2
iou_threshold = 0.5
swap_order = True
activation = 'sigmoid'

s_min = 0.2
s_max = 0.95
aspect_ratios = (1.0, 2.0, 0.5, 3, 1.0 / 3)
interpolated_scale_aspect_ratio = 1.0
feature_map_sizes = [20, 10, 5, 3, 2, 1]
num_layers = len(feature_map_sizes)

scales = [s_min + (s_max - s_min) * i / (num_layers - 1) for i in range(num_layers)] + [1.0]
box_specs_list = []
for layer, scale, scale_next in zip(
        range(num_layers), scales[:-1], scales[1:]):
    layer_box_specs = []
    if layer == 0:
        layer_box_specs = [(0.1, 1.0), (scale, 2.0), (scale, 0.5)]
    else:
        for aspect_ratio in aspect_ratios:
            layer_box_specs.append((scale, aspect_ratio))
        if interpolated_scale_aspect_ratio > 0.0:
            layer_box_specs.append((np.sqrt(scale * scale_next),
                                    interpolated_scale_aspect_ratio))
    box_specs_list.append(layer_box_specs)

priors = generate_ssd_priors(box_specs_list, feature_map_sizes)