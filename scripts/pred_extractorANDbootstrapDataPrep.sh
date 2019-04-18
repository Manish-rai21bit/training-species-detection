# Run for a single TF Record
# ----- Extracts the predictions from the TR Records into a CSV ----- #
python predictorExtractor_main.py \
    --tfrecord_path_list '/home/ubuntu/data/tensorflow/my_workspace/training_demo/Predictions/snapshot_serengeti_s01_s06-10000-20000.record' \
    --output_csv '/home/ubuntu/data/tensorflow/my_workspace/training_demo/Predictions/snapshot_serengeti_s01_s06-10000-20000.csv'

# ----- Creates a consolidated file with the outer join of prediction and groundtruth pecies count ----- #
python prediction_groundtruth_consolidation_main.py \
    --prediction_csv_path '/home/ubuntu/data/tensorflow/my_workspace/training_demo/Predictions/snapshot_serengeti_s01_s06-10000-20000.csv'  \
    --groundtruth_csv_path '/home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/data/LILA/db_export_season_all_cleaned.csv' \
    --label_map_json '/home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/data/LILA/label_map.json' \
    --outfile '/home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/pred_groundtruth_consolidate_csv_s01_s06-10000-20000.csv'

# ----- Creates a CSV with only the correct predictions ----- #
python bootstrapping_data_prep_main.py \
    --pred_groundtruth_consolidate_csv '/home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/pred_groundtruth_consolidate_csv_s01_s06-10000-20000.csv' \
    --prediction_csv_path '/home/ubuntu/data/tensorflow/my_workspace/training_demo/Predictions/snapshot_serengeti_s01_s06-10000-20000.csv' \
    --label_map_json '/home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/data/LILA/label_map.json' \
    --outfile '/home/ubuntu/data/tensorflow/my_workspace/training_demo/Predictions/bootstrap_data_snapshot_serengeti_s01_s06-10000-20000.csv'

# ----- What this does is creates the TF record that is ready to be fed into model training ----- #
# This part needs change - look up in the already existing TF Record for the image encodings.#
python tfr_encoder_for_bootstarpping_main.py \
    --image_filepath '/panfs/roc/groups/5/packerc/shared/albums/SER/' \
    --bounding_box_csv 'bootstrap_data_snapshot_serengeti_s01_s06-10000-20000.csv' \
    --label_map_json '/home/packerc/rai00007/camera-trap-detection/data/LILA/label_map.json' \
    --output_tfrecord_file 'out_tfr_file-10000-20000.record'


# -------------------------------------------------------------- #
# -------- Run the pipeline mentioned above in a loop ---------- #
# -------------------------------------------------------------- #

cd /home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection
mkdir /home/ubuntu/data/tensorflow/my_workspace/training_demo/Predictions/round1/

for entry in "/home/ubuntu/species_detection/test_eco1"/*
do
    tfr_filename=$(basename $entry)
    # ----- Extracts the predictions from the TR Records into a CSV ----- #
    python predictorExtractor_main.py \
        --tfrecord_path_list $entry \
        --output_csv '/home/ubuntu/data/tensorflow/my_workspace/training_demo/Predictions/round1/'${tfr_filename/'.record'}'.csv' \
        --groundtruth_csv_path '/home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/data/LILA/db_export_season_all_cleaned.csv' \
        --label_map_json '/home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/data/LILA/label_map.json' \
        --score_threshold 0.75;

    # ----- Creates a consolidated file with the outer join of prediction and groundtruth pecies count ----- #
    python prediction_groundtruth_consolidation_main.py \
        --prediction_csv_path '/home/ubuntu/data/tensorflow/my_workspace/training_demo/Predictions/round1/'${tfr_filename/'.record'}'.csv'  \
        --groundtruth_csv_path '/home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/data/LILA/db_export_season_all_cleaned.csv' \
        --label_map_json '/home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/data/LILA/label_map.json' \
        --outfile '/home/ubuntu/data/tensorflow/my_workspace/training_demo/Predictions/round1/pred_groundtruth_consolidate_'${tfr_filename/'.record'}'.csv' ;

    # ----- Creates a CSV with only the correct predictions ----- #
    python bootstrapping_data_prep_main.py \
        --pred_groundtruth_consolidate_csv '/home/ubuntu/data/tensorflow/my_workspace/training_demo/Predictions/round1/pred_groundtruth_consolidate_'${tfr_filename/'.record'}'.csv' \
        --prediction_csv_path '/home/ubuntu/data/tensorflow/my_workspace/training_demo/Predictions/round1/'${tfr_filename/'.record'}'.csv'  \
        --label_map_json '/home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/data/LILA/label_map.json' \
        --outfile '/home/ubuntu/data/tensorflow/my_workspace/training_demo/Predictions/round1/bootstrap_data_'${tfr_filename/'.record'}'.csv' ;

    # ----- What this does is creates the TF record that is ready to be fed into model training ----- #
done

for entry in "/home/packerc/rai00007/round1/input/"*
do
    tfr_filename=$(basename $entry)
    python tfr_encoder_for_bootstarpping_main.py \
        --image_filepath '/panfs/roc/groups/5/packerc/shared/albums/SER/' \
        --bounding_box_csv $entry \
        --label_map_json '/home/packerc/rai00007/camera-trap-detection/data/LILA/label_map.json' \
        --output_tfrecord_file '/home/packerc/rai00007/round1/output_tfr/bootstrap_data_round1'${tfr_filename/'.csv'/'.record'}
done

imgpath='/home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/test_images0.90'
mkdir $imgpath
python tfr_visualization_main.py \
 --filename_list '/home/ubuntu/species_detection/test_eco1/snapshot_serengeti_s01_s06_0_199999.record-00097-of-00100' \
 --outfile $imgpath \
 --label_map_json '/home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/data/LILA/label_map.json' \
 --num_batches 128 \
 --score_threshold 0.90


for entry in "/home/ubuntu/data/tensorflow/my_workspace/training_demo/Predictions/round1/snapshot_serengeti_s01_s06_"*
do
    python intersection_check.py \
    --image_filepath=$entry
done



### running this after fixing the bootstrapping_data_prep.py
### This function was not updating the species label which was fixed.
for entry in "/home/ubuntu/species_detection/test_eco1"/*
do
    tfr_filename=$(basename $entry)
    # ----- Creates a CSV with only the correct predictions ----- #
    python bootstrapping_data_prep_main.py \
        --pred_groundtruth_consolidate_csv '/home/ubuntu/data/tensorflow/my_workspace/training_demo/Predictions/round1/pred_groundtruth_consolidate_'${tfr_filename/'.record'}'.csv' \
        --prediction_csv_path '/home/ubuntu/data/tensorflow/my_workspace/training_demo/Predictions/round1/'${tfr_filename/'.record'}'.csv'  \
        --label_map_json '/home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/data/LILA/label_map.json' \
        --outfile '/home/ubuntu/data/tensorflow/my_workspace/training_demo/Predictions/round1/bootstrap_data_'${tfr_filename/'.record'}'.csv' ;

    # ----- What this does is creates the TF record that is ready to be fed into model training ----- #
done
