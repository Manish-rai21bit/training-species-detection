sudo mount /dev/xvdf ~/data
export LC_ALL=C
# cd ~
source ~/test_env/bin/activate
export PYTHONPATH="${PYTHONPATH}:/home/ubuntu/data/tensorflow/models/"
export PYTHONPATH="${PYTHONPATH}:/home/ubuntu/data/tensorflow/models/research"
export PYTHONPATH="${PYTHONPATH}:/home/ubuntu/data/tensorflow/models/research/slim/"

cd ~/data/tensorflow/my_workspace/training_demo/
