"""
A script to encode images to TF Record for testing purposes.

Usage:
Do include the tensorflow directories in PYTHONPATH using:
export PYTHONPATH="${PYTHONPATH}:/home/packerc/rai00007/tensorflow/models/"
export PYTHONPATH="${PYTHONPATH}:/home/packerc/rai00007/tensorflow/models/research"
export PYTHONPATH="${PYTHONPATH}:/home/packerc/rai00007/tensorflow/models/research/slim/"

python dataset_tools/create_test_tf_record.py \
    --image_list_csv "./Data/dataset_for_testing/msi_snapshot_serengeti.csv" \
    --output_tfrecord_file "./test.record" \
    --num_shards 1
"""

import os, sys, csv, argparse
import tensorflow as tf
import object_detection.utils.dataset_util as dataset_util
from PIL import ImageFile
# imports for sharding
import contextlib2
from object_detection.dataset_tools import tf_record_creation_util

ImageFile.LOAD_TRUNCATED_IMAGES = True

"""This function reads a raw image, resizes it with aspect ratio preservation and returns the byte string"""
from PIL import Image
import numpy as np
import io

def resize_jpeg(image,  max_side):
    """ Take Raw JPEG resize with aspect ratio preservation
         and return bytes
    """
    img = Image.open(image)
    img.thumbnail([max_side, max_side], Image.ANTIALIAS)
    b = io.BytesIO()
    img.save(b, 'JPEG')
    image_bytes = b.getvalue()
    return image_bytes


""" This function creates a tfrecord example from the dictionary element!"""
def create_tf_example(data_dict):
    #with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
    encoded_jpg = resize_jpeg(os.path.join('/panfs/roc/groups/5/packerc/shared/albums/SER/', data_dict) + '.JPG',  1000)
    #encoded_jpg_io = io.BytesIO(encoded_jpg)
    #image = Image.open(encoded_jpg_io)
    #width, height = image.size
    filename = data_dict.encode('utf-8')
    image_format = b'jpg'

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/filename': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format)
    }))


    return tf_example

"""This iterates over each dictionary item, creates tf examples,
    serializes the tfrecord examples and writes to a tfrecord file!!!
    As of now, it saves the TFRecord file in the home directory where the code is executed"""
def encode_to_tfr_record(test_feature, out_tfr_file, num_shards=1):
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, out_tfr_file, num_shards)
        for index, example in enumerate(test_feature):
            tf_example = create_tf_example(example)
            output_shard_index = index % num_shards
            output_tfrecords[output_shard_index].write(tf_example.SerializeToString())


def main(image_list_csv, output_tfrecord_file, num_shards):
    with open(image_list_csv,'r') as f:
        l = []
        rd = csv.reader(f)
        for val in rd:
            l.append(val)

    event_dict = l[0]
    encode_to_tfr_record(event_dict, output_tfrecord_file, num_shards)

if  __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_list_csv", type=str, required=True,
        help="path to the list containing the images"
        )
    parser.add_argument(
        "--output_tfrecord_file", type=str, required=True,
        help="path to the TF Records containing the encoded images"
        )
    parser.add_argument(
        "--num_shards", type=int, required=False, default=1,
        help="Number of shard to create"
        )

    args = parser.parse_args()

    main(args.image_list_csv, args.output_tfrecord_file, args.num_shards)
