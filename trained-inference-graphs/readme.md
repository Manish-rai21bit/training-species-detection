Directory contains the trained graph for species detection. The graph can be exported from the trained model using the script below:

```
python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path training/faster_rcnn_resnet101_coco.config \
    --trained_checkpoint_prefix training/model.ckpt-<checkpoint number> \
    --output_directory trained-inference-graphs/output_inference_graph
```
