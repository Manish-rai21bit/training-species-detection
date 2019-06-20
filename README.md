# Training Species Detection
Repository for:
1. Training species detection model.
2. Holding trained graph.
3. Holding predictions.

## Architecture for Detection
Faster R-CNN architecture pre-trained on the COCO dataset was used to initialise the fine-tuning of the graph.  

## Dataset
**Initial Training** - Snapshot Serengeti data from the [LILA Repository](http://lila.science/datasets/snapshot-serengeti) (Season 1 - Season 6) was used for initial training. 38,000 images of 48 different species was used for this round of training.


**Bootstrapping Round Training** - More training dataset were generated through a weak supervision. We call this bootstrapping procedure. Data from Season 1 - Season 6 from Snapshot Serengeti was used to generate training dataset using the bootstrapping procedure.


**Test Datasets** -
- Detection was tested on images with bounding box level annotations draws from dataset created by [Schneider.et.al](https://dataverse.scholarsportal.info/dataset.xhtml?persistentId=doi:10.5683/SP/TPB5ID).
- Classification, count and generalisation performance was evaluated of Snapshot Serengeti Season 10 dataset. This dataset has image level classification annotations.

## Evaluation
- COCO detection metrics and Pascal VOC detection metrics can be used for evaluation of detections.
- Presence and counts of animals in the image was evaluated by using classification metrics like Precision and Recall:
    - Classification performance - if the animal is present in the image or not.
    - Count performance - Count of the boxes predicted by the model is within a specified deviance from the ground truth counts.

# Usage:
1. Training Species Detection Model:
  1. Data Preparation (TF Records encoder from CSV) - Training the model from the tensorflow's object_detection_api can be done faster with dataset encoded as TF Records.

  2.  
