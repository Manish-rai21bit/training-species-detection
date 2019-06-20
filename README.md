# Training Species Detection
Repository for:
1. Training species detection model.
2. Holding trained graph.
3. Holding predictions.

## Architecture for Detection
Faster R-CNN architecture pre-trained on the COCO dataset was used to initialize the fine-tuning of the graph.  

## Dataset
**Initial Training** - Snapshot Serengeti data from the [LILA Repository](http://lila.science/datasets/snapshot-serengeti) (Season 1 - Season 6) was used for initial training. 38,000 images of 48 different species was used for this round of training. \

**Bootstrapping Round Training** - More training dataset were generated through a weak supervison. We call this bootstrapping procedure. Data from Season 1 - Season 6 from Snapshot serengeti was used to generate training dataset using the bootstrapping procedure. 
