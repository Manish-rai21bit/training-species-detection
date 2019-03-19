Directory for:
1. storing evaluation pbtxt and config files
2. storing evaluation outputs


Sample of how to create the config files and call the evaluation:


```
# Definig the molel config file
echo "
faster_rcnn {
    num_classes: 48
    number_of_stages: 2
    image_resizer {
      keep_aspect_ratio_resizer {
        min_dimension: 600
        max_dimension: 1024
      }
    }
    feature_extractor {
      type: 'faster_rcnn_resnet101'
      first_stage_features_stride: 16
    }
    first_stage_anchor_generator {
      grid_anchor_generator {
        scales: [0.25, 0.5, 1.0, 2.0]
        aspect_ratios: [0.5, 1.0, 2.0]
        height_stride: 16
        width_stride: 16
      }
    }
    first_stage_box_predictor_conv_hyperparams {
      op: CONV
      regularizer {
        l2_regularizer {
          weight: 0.0
        }
      }
      initializer {
        truncated_normal_initializer {
          stddev: 0.01
        }
      }
    }
    first_stage_nms_score_threshold: 0.0
    first_stage_nms_iou_threshold: 0.7
    first_stage_max_proposals: 32
    first_stage_localization_loss_weight: 2.0
    first_stage_objectness_loss_weight: 1.0
    initial_crop_size: 14
    maxpool_kernel_size: 2
    maxpool_stride: 2
    second_stage_batch_size: 32
    second_stage_box_predictor {
      mask_rcnn_box_predictor {
        use_dropout: false
        dropout_keep_probability: 1.0
        fc_hyperparams {
          op: FC
          regularizer {
            l2_regularizer {
              weight: 0.0
            }
          }
          initializer {
            variance_scaling_initializer {
              factor: 1.0
              uniform: true
              mode: FAN_AVG
            }
          }
        }
      }
    }
    second_stage_post_processing {
      batch_non_max_suppression {
        score_threshold: 0.0
        iou_threshold: 0.6
        max_detections_per_class: 75
        max_total_detections: 75
      }
      score_converter: SOFTMAX
    }
    second_stage_localization_loss_weight: 2.0
    second_stage_classification_loss_weight: 1.0
  }
" > /home/ubuntu/data/tensorflow/my_workspace/training_demo/test_eval_metrics/model_config.pbtxt

echo "
label_map_path: '/home/ubuntu/data/tensorflow/my_workspace/training_demo/annotations/label_map_lila.pbtxt'
tf_record_input_reader: { input_path: '/home/ubuntu/data/tensorflow/my_workspace/training_demo/annotations/test_schneider.record' }
shuffle:false
" > /home/ubuntu/data/tensorflow/my_workspace/training_demo/test_eval_metrics/test_input_config.pbtxt

## eval_evaluation_config.pbtxt - for evaluation on evaluation set
## train_evaluation_config.pbtxt - for evaluation on evaluation set


echo "
  num_examples: 3964
  # 3964
  # Note: The below line limits the evaluation process to 10 evaluations.
  # Remove the below line to evaluate indefinitely.
  # Detection metrics - coco_detection_metrics, pascal_voc_detection_metrics
  max_evals: 1
  metrics_set: 'coco_detection_metrics'
  include_metrics_per_category : True
  " > /home/ubuntu/data/tensorflow/my_workspace/training_demo/test_eval_metrics/eval_config.pbtxt

python eval.py \
    --logtostderr \
    --checkpoint_dir=trained-inference-graphs/output_inference_graph/ \
    --eval_dir=test_eval_metrics \
    --eval_config_path='/home/ubuntu/data/tensorflow/my_workspace/training_demo/test_eval_metrics/eval_config.pbtxt' \
    --input_config_path='/home/ubuntu/data/tensorflow/my_workspace/training_demo/test_eval_metrics/test_input_config.pbtxt' \
    --model_config_path='/home/ubuntu/data/tensorflow/my_workspace/training_demo/test_eval_metrics/model_config.pbtxt'
```
