Directory containing:
1. last 5 models and checkpoints
2. configuration file for model training


Model training can be started using th script:

```
cd ~/data/training-species-detection/

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
