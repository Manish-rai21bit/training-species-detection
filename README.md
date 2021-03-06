# Training Species Detection
**Detecting Animals and getting species level counts**

This repository contains procedures and codes for training an object detection architecture on wildlife camera-trap images. It uses transfer learning to finetune a pre-trained model like SSD, YOLO, or Faster R-CNN. For this particular work, I have trained a Faster R-CNN architecture with Resnet-101 encodings.


Repository for:
1. Training species detection model
2. Creating tensorflow graph with trained model
3. Making predictions

## Installations:
1. Python3 (https://www.python.org)
2. tensorflow 1.9+ (I've build this in tensorflow 1.9) (https://www.tensorflow.org)
3. tensorflow's [object_detection_api](https://github.com/tensorflow/models/tree/master/research/object_detection) - clone the repository and add it to your local PYTHONPATH:

```
export PYTHONPATH="${PYTHONPATH}:<path to cloned tensorflow object detection api>/tensorflow/models/"
export PYTHONPATH="${PYTHONPATH}:<path to cloned tensorflow object detection api>/tensorflow/models/research"
export PYTHONPATH="${PYTHONPATH}:<path to cloned tensorflow object detection api>/tensorflow/models/research/slim/"
```
4. Dependent installations for tensorflow's [object_detection_api](https://github.com/tensorflow/models/tree/master/research/object_detection) should be installed by following the instructions mentioned [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)  
5. Clone this [repository](https://github.com/Manish-rai21bit/training-species-detection)
6. Clone and add this repository, [camera-trap-detection](https://github.com/Manish-rai21bit/camera-trap-detection.git), to the PYTHONPATH

```
export PYTHONPATH="${PYTHONPATH}:<path to cloned camera-trap-detection api>"
```

## Architecture for Detection
Faster R-CNN architecture pre-trained on the COCO dataset was used to initialise and fine-tuning of the graph.  

## Dataset
**Initial Training** - Snapshot Serengeti data from the [LILA Repository](http://lila.science/datasets/snapshot-serengeti) (Season 1 - Season 6) was used for initial training. 38,000 images of 48 different species was used for this round of training.


**Bootstrapping Round Training** - More training dataset were generated through a weak supervision. We call this bootstrapping procedure. Data from Season 1 - Season 6 from Snapshot Serengeti was used to generate training dataset using the bootstrapping procedure. The codes for bootstrapping can be found in this repo [camera-trap-detection](https://github.com/Manish-rai21bit/camera-trap-detection)


**Test Datasets** -
- Detection was tested on images with bounding box level annotations draws from dataset created by [Schneider.et.al](https://dataverse.scholarsportal.info/dataset.xhtml?persistentId=doi:10.5683/SP/TPB5ID).
- Classification, count and generalisation performance was evaluated of Snapshot Serengeti Season 10 dataset. This dataset has image level classification annotations.

## Evaluation
- COCO detection metrics and Pascal VOC detection metrics can be used for evaluation of detections.
- Presence and counts of animals in the image was evaluated by using classification metrics like Precision and Recall:
    - Classification performance - if the animal is present in the image or not.
    - Count performance - Count of the boxes predicted by the model is within a specified deviance from the ground truth counts.

# Usage:
The directory structure that I am using for model training, evaluation, graph export and prediction is same as this directory. Each directory and sub-directory is used for:
```
training-species-detection
  -- annotations - Data used for training - TF Records for training and evaluation and label maps
  -- pre-trained-model - holds the pre-trained model used for Transfer Learning
  -- trained-inference-graphs - Directory where the trained model graph is exported
  -- test_eval_metrics - Stores the evaluation measures
  -- training - Used during model training to store intermediate checkpoints, configuration files, etc.
  -- Predictions - Stores predictions
```


1. Training Species Detection Model:

  - **Data Preparation** (TF Records encoder from CSV) - Training the model from the tensorflow's object_detection_api can be done faster with dataset encoded as TF Records. TF Records can be created by the Script below:

  ```
  python create_training_tf_record_main.py \
      --image_filepath '/panfs/roc/groups/5/packerc/shared/albums/SER/' \
      --bounding_box_csv '/home/packerc/rai00007/camera-trap-detection/bounding_box_test_dataset.csv' \
      --label_map_json '/home/packerc/rai00007/camera-trap-detection/data/LILA/label_map.json' \
      --output_tfrecord_file 'annotations/TestTFR/out_tfr_file.record' \
      --num_shards 5
  ```

  The Raw images from Snapshot Serengeti are on MSI and so I suggest to do this step on the MSI and move the TF Records to the environment where model training is happening. In my case, my model is being trained and evaluated on AWS and so I move the TF Records to the AWS through scp file transfer using command similar as below:

  ```
  scp -i ~/MyKeyPair.pem  /panfs/roc/groups/5/packerc/rai00007/my_workspace/training-species-detection/annotations/TestTFR/*  ubuntu@ec2-18-221-73-140.us-east-2.compute.amazonaws.com:/home/ubuntu/species_detection/my_workspace/training-species-detection/annotations/TestTFR/
  ```

  - **Training the object detector** using the Tensorflow's Object Detection API - In this step we use the TF Records created above to fine-tune a pre-trained model to detect animals from the Serengeti National Park. Transfer Learning can be used on any other location as well. Parameters that need changing before trianing starts:

The configuration file in the 'training' directory needs to be updated with the filepath of training and evaluation TF Records:


    ```
    train_input_reader: {
    tf_record_input_reader {
        input_path: "****** Training TF Records *******"
      }
      label_map_path: "****** Evaluation TF Records *******"
    }

    eval_config: {
      num_examples: 3964
      # Note: The below line limits the evaluation process to 10 evaluations.
      # Remove the below line to evaluate indefinitely.
      max_evals: 10
      metrics_set: "coco_detection_metrics"
    }
    ```

The below script can be used to initiate the model training:

    ```
    cd training-species-detection/

    PIPELINE_CONFIG_PATH='training/faster_rcnn_resnet101_coco.config'
    MODEL_DIR='./training/'
    NUM_TRAIN_STEPS=900000
    NUM_EVAL_STEPS=1500

    python model_main.py \
        --pipeline_config_path=${PIPELINE_CONFIG_PATH}  \
        --model_dir=${MODEL_DIR} \
        --num_train_steps=${NUM_TRAIN_STEPS} \
        --num_eval_steps=${NUM_EVAL_STEPS} \
        --alsologtostderr
    ```

The training process can be monitored on tensorboard. If you are on AWS and want to create a tunnel/forward port to local then you can use this script:

    ```
    tensorboard --logdir=training --host localhost --port=6006
    ```

On a different shell:
    ```
    ssh -N -L 6006:localhost:6006 -i ~/.ssh/MyKeyPair.pem ubuntu@ec2-18-221-73-140.us-east-2.compute.amazonaws.com
    ```
2. **Graph Export**: The training process will store the latest 5 checkpoints that can be exported and used for predictions on incoming images. A trained checkpoint can be exported using the script:


Before we can export the trained graph, we'll need to remove the directory trained-inference-graphs/output_inference_graph
```
rm -r trained-inference-graphs/output_inference_graph
```

```
python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path training/faster_rcnn_resnet101_coco.config \
    --trained_checkpoint_prefix training/model.ckpt-<checkpoint number> \
    --output_directory trained-inference-graphs/output_inference_graph
```

3. **Making Predictions**: One of the easiest way to visualise predictions on a few images is using the off the shelf notebook available [here](https://github.com/Manish-rai21bit/training-species-detection/blob/master/object_detection_tutorial.ipynb)

Apart from this, in case we could make predictions on a large number of images more efficiently. For this to happen, we first need to convert the raw images into TF Records using the [script](https://github.com/Manish-rai21bit/training-species-detection/blob/master/dataset_tools/create_test_tf_record.py). Detailed instruction can for running this script can be found in the code's doc string.

Once we have the images encoded as TF records on which we want to run predictions, the below script can be used to make predictions:

```
python <path to tensorflow/models/research/>object_detection/inference/infer_detections.py \
  --input_tfrecord_paths=<path to input TF record, test.record-00000-of-00001> \
  --output_tfrecord_path=<'path to the output TF records, test_output.record '> \
  --inference_graph=<'path to the trained graph, i.e - trained-inference-graphs/output_inference_graph/frozen_inference_graph.pb'> \
  --discard_image_pixels
```

4. **Visualising Predictions**:

```
python /home/packerc/rai00007/camera-trap-detection/tfr_visualization_main.py \
    --filename_list 'test_output.record' \
    --outfile 'Data/images/visual_img/' \
    --label_map_json '/home/packerc/rai00007/camera-trap-detection/data/LILA/label_map.json' \
    --num_batches 4 \
    --TFRecord_type 'Pred' \
    --score_threshold 0.5
```
5. **Decoding Prediction TF Records**:
After all this, we'd want to export the predictions in a nice CSV and the scripts below will help extract the TF Records with predictions into a CSV format. The csv contains the following [columns](https://github.com/Manish-rai21bit/training-species-detection/blob/master/Predictions/test.csv)


```
python predictorExtractor_main.py \
    --tfrecord_path_list 'test_output.record' \
    --output_csv '/home/ubuntu/species_detection/my_workspace/training-species-detection/Predictions/test.csv' \
    --label_map_json '/home/ubuntu/data/tensorflow/my_workspace/camera-trap-detection/data/LILA/label_map.json' \
    --score_threshold 0.5
```
