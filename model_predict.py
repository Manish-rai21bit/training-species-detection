""" This module is for prediction. """

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
#  imports from the object detection module.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util



# What model to download.
MODEL_NAME = '/home/ubuntu/data/tensorflow/my_workspace/training_demo/trained-inference-graphs/output_inference_graph'#'ssd_mobilenet_v1_coco_2017_11_17'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/home/ubuntu/data/tensorflow/my_workspace/training_demo/annotations/', 'label_map.pbtxt')
NUM_CLASSES = 46

PATH_TO_TEST_IMAGES_DIR = '/home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 9) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 12)



# Load a (frozen) Tensorflow model into memory.
def load_tf_graph(PATH_TO_FROZEN_GRAPH):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            return tf.import_graph_def(od_graph_def, name='')

def load_label_map(PATH_TO_LABELS, NUM_CLASSES):
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return label_map, categories, category_index


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.1), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def get_per_image_prediction(output_dict, image, threshold_detection_score=0.5):
    image_np = load_image_into_numpy_array(image)
    label_map, categories, category_index = load_label_map(PATH_TO_LABELS, NUM_CLASSES)
    width = image_np.shape[1]  # Number of columns
    height = image_np.shape[0]  # number of rows
    category_index = label_map_util.create_category_index(categories)

    image_bb_prediction = []

    for i in range(len(output_dict['detection_boxes'])):
        bb_prediction = {}
        if output_dict['detection_scores'][i] > threshold_detection_score:
            # class_name = category_index[output_dict['detection_classes'][i]]['name']
            bb_prediction['class_name'] = category_index[output_dict['detection_classes'][i]]['name']
            bb_prediction['detection_scores'] = output_dict['detection_scores'][i]
            bb_prediction['detection_classes'] = output_dict['detection_classes'][i]
            bb_prediction['bb_xmin'] = int(width * output_dict['detection_boxes'][i][1])
            bb_prediction['bb_ymin'] = int(height * output_dict['detection_boxes'][i][0])
            bb_prediction['bb_xmax'] = int(width * output_dict['detection_boxes'][i][3])
            bb_prediction['bb_ymax'] = int(height * output_dict['detection_boxes'][i][2])
            bb_prediction['width'] = width
            bb_prediction['height'] = height
            image_bb_prediction.append(bb_prediction)
    return image_bb_prediction

        # print("{class: %s, prediction: %s, boundingbox: %s,%i,%i,%i,%i,%i,%i,%i}"
        #       % (class_name,
        #          output_dict['detection_scores'][i],
        #          TEST_IMAGE_PATHS[2],
        #          width,
        #          height,
        #          output_dict['detection_classes'][i],
        #          int(width * output_dict['detection_boxes'][i][1]),  # The boxes are given normalized and in row/col order
        #          int(height * output_dict['detection_boxes'][i][0]),
        #          int(width * output_dict['detection_boxes'][i][3]),
        #          int(height * output_dict['detection_boxes'][i][2])
        #             ))

