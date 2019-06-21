"""
This module contains a set of functions for creating TFRecord files with box leval annotations.
The outputs of this module can be fed into training the Tensorflow's object_detection_api.
Contains TFRecord encoder and decored and supporting functions for dealing with TFRecord.


Example usage:

python tfr_encoder_for_bootstarpping_main.py \
    --image_filepath '/panfs/roc/groups/5/packerc/shared/albums/SER/' \
    --bounding_box_csv 'bootstrap_data_snapshot_serengeti_s01_s06-10000-20000.csv' \
    --label_map_json '/home/packerc/rai00007/camera-trap-detection/data/LILA/label_map.json' \
    --output_tfrecord_file 'out_tfr_file-10000-20000.record'


Example of CSV format to be used is:
filename,class,xmin,ymin,xmax,ymax
S4/F13/F13_R1/S4_F13_R1_IMAG0526,zebra,0.0,0.38,0.12,0.54
S4/F13/F13_R1/S4_F13_R1_IMAG0526,zebra,0.19,0.32,0.56,0.66
S4/S13/S13_R2/S4_S13_R2_IMAG0040,gazelleThomsons,0.05,0.54,0.12,0.62
S4/S13/S13_R2/S4_S13_R2_IMAG0040,gazelleThomsons,0.56,0.48,1.0,1.0


This function takes the csv datasets in the form mentioned above and creates a TFRecord file.
The functions included are:
1. csvtodict - creates dictionary objects. have to rename this function.
2. create_tf_example - Creates a tf_example.
3. encode_to_tfr_record - Creates a TF Record file
"""

import tensorflow as tf
import os, csv, io
from PIL import Image, ImageFile
# imports for sharding
import contextlib2
from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util

import dataset_tools.data_prep_utils as dataprep_utils
import dataset_tools.image as img

ImageFile.LOAD_TRUNCATED_IMAGES = True


def csvtodict(image_filepath, bb_data):
    """
    Converts the CSV to dictionary object.
    Inputs:
    image_filepath - file directory for the images
    bb_data - path to the csv

    Return:
    record_dict - dictionary object
    """
    lst = []
    record_dict = {}
    csvfile = open(os.path.join(bb_data), 'r')
    csvdata = csv.reader(csvfile, delimiter=',')
    first_row = next(csvdata)
    for row in csvdata:
        if row[0] not in record_dict:
            record_dict[row[0]] = {'metadata' : {"SiteID": row[0].split('/')[1],
                                  "DateTime": "placeholder",
                                  "Season": row[0].split('/')[0]},
                                    'images' : [{"Path" : os.path.join(image_filepath, row[0] + '.JPG'), #points to the route of image on the disk
                                "URL" : 'placeholder',
                                "dim_x" : 'placeholder',
                                "dim_y" : 'placeholder',
                                "image_label" : "tbd", # This is the primary label in case we want to have some for the whole image
                                'observations' : []
                               }]
                                    }
        record_dict[row[0]]['images'][0]['observations'].append({'bb_ymin': row[3],
                                                   'bb_ymax': row[5],
                                                      'bb_primary_label': row[1],
                                                      'bb_xmin': row[2],
                                                      'bb_xmax': row[4],
                                                      'bb_label': {"species" : row[1],
                                                    "pose" : "standing/ sitting/ running"
                                                }})
    return record_dict

def create_tf_example(data_dict,
                      label_map
                     ):
    """
    This function creates a tfrecord example from the dictionary element!
    """
    encoded_jpg = img.resize_jpeg((data_dict['images'][0]['Path']),  1000)
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    width = int(width)
    height = int(height)

    filename = data_dict['images'][0]['Path'].encode('utf-8')
    image_format = b'jpg'
    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    classes_text, classes = [], []

    for bb_record in data_dict['images'][0]['observations']:
        xmins.append(float(bb_record['bb_xmin']))
        xmaxs.append(float(bb_record['bb_xmax']))
        ymins.append(float(bb_record['bb_ymin']))
        ymaxs.append(float(bb_record['bb_ymax']))
        classes_text.append(bb_record['bb_primary_label'].encode('utf8'))
        classes.append(label_map[bb_record['bb_primary_label']])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def encode_to_tfr_record(bounding_box_dict, label_map, out_tfr_file, num_shards=1):
    """
    This iterates over each dictionary item, creates tf examples,
    serializes the tfrecord examples and writes to a tfrecord file!!!
    As of now, it saves the TFRecord file in the home directory where the code is executed
    """
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, out_tfr_file, num_shards
        )
        index = 0
        for k, v in bounding_box_dict.items():
            if index%1000==0:
                print("Processing image number {0}".format(index))
            tf_example = create_tf_example(v, label_map)
            output_shard_index = index % num_shards
            output_tfrecords[output_shard_index].write(tf_example.SerializeToString())
            index+=1
