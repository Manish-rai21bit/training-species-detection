################################################################################
##### Main function to extract the predictions from a TFRecord into a CSV  #####
################################################################################
"""
This function is run using the script:
python predictorExtractor_main.py \
    --tfrecord_path_list 'test_output.record' \
    --output_csv '/home/ubuntu/species_detection/my_workspace/training-species-detection/Predictions/test.csv' \
    --label_map_json '/home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/data/LILA/label_map.json' \
    --score_threshold 0.5;
"""

import argparse
import pandas as pd

from dataset_tools.predictor_extractor import predictorExtractor
import dataset_tools.data_prep_utils as dataprep_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tfrecord_path_list", nargs='+', type=str, required=True,
        help="Path to TFRecord files")
    parser.add_argument(
        "--output_csv", type=str, required=True,
        help="output csv file"
        )
    parser.add_argument(
        "--batch_size", type=int, default=512,
        help="batch size")
    parser.add_argument(
        "--score_threshold", type=float, default=0.5,
        help="score thresholds to write to csv")
    parser.add_argument(
        "--discard_image_pixels", type=bool, default=True,
        help="True to discard the pixel encodings or when pixel encodings are not present in the datafile")
    parser.add_argument(
        "--label_map_json", type=str, required=True,
        help="path to the label map json file")

    args = parser.parse_args()

    label_map = dataprep_utils.get_label_map_from_json(args.label_map_json)
    label_map_rev =  {v:k for k, v in label_map.items()}

    predictorExtractor(args.tfrecord_path_list, 
                       args.output_csv, 
                       label_map_rev, 
                       score_threshold=args.score_threshold,
                       batch_size=args.batch_size,
                       discard_image_pixels=args.discard_image_pixels
                      )
