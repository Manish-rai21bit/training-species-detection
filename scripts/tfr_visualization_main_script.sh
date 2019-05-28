############################
### plot the predictions ###
############################
mkdir /home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/test_images_s10/round1

imgpath='/home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/test_images_s1_s6/round3_addition/herd'
# mkdir $imgpath
python tfr_visualization_main.py \
 --filename_list '/home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/data/bootstrapping/round3_record_all/herd_bbox.record' \
 --outfile $imgpath \
 --label_map_json '/home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/data/LILA/label_map.json' \
 --num_batches 128 \
 --score_threshold 0 \
 --TFRecord_type 'Pred'


############################
# Plot the training images #
############################
outfile='/home/ubuntu/species_detection/data/test_img_singleSpecies/'
python tfr_visualization_main.py \
   --filename_list '/home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/data/bootstrapping/bootstrap_LowerBound50p/round1_record_all/singleSpecies_bbox.record' \
   --outfile $outfile \
   --label_map_json '/home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/data/LILA/label_map.json' \
   --num_batches 128 \
   --TFRecord_type 'Train'


############################
### Plot the raw images. ###
############################

outfile='/home/ubuntu/species_detection/data/test_img_raw/'
python tfr_visualization_main.py \
   --filename_list '/home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/data/test_snapshot_serengeti_s10/encoded_images_for_test/msi_test_image_list_s10.record-00000-of-00100'\
   --outfile $outfile \
   --label_map_json '/home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/data/LILA/label_map.json' \
   --num_batches 2 \
   --TFRecord_type 'Raw'

