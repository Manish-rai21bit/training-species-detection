"""
The main function for create_training_tf_record.py

python create_training_tf_record_main.py \
    --image_filepath '/panfs/roc/groups/5/packerc/shared/albums/SER/' \
    --bounding_box_csv 'bootstrap_data_snapshot_serengeti_s01_s06-0-10000.csv' \
    --label_map_json '/home/packerc/rai00007/camera-trap-detection/data/LILA/label_map.json' \
    --output_tfrecord_file 'out_tfr_file.record' \
    --num_shards 1
"""

import argparse

import data_prep.data_prep_utils as dataprep_utils
import data_prep.create_training_tf_record as tfr

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--image_filepath", type=str, required=True,
    help="path to the image file")
    parser.add_argument(
    "--bounding_box_csv", type=str, required=True,
    help="csv with bounding boxes")
    parser.add_argument(
    "--label_map_json", type=str, required=True,
    help="label map json")
    parser.add_argument(
    "--output_tfrecord_file", type=str, required=True,
    help="output path for TFRecord")
    parser.add_argument(
    "--num_shards", type=int, required=False,
    help="number of shards for the output TF record")

    args = parser.parse_args()

    bounding_box_dict = tfr.csvtodict(args.image_filepath, args.bounding_box_csv)
    label_map = dataprep_utils.get_label_map_from_json(args.label_map_json)
    tfr.encode_to_tfr_record(bounding_box_dict, label_map, args.output_tfrecord_file, args.num_shards)
