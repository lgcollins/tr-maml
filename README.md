# Task-Robust Model-Agnostic Meta-Learning
This repository contains the code used to run the few-shot sinusoid regression and image classification experiments discussed in the TR-MAML paper. For each of these sets of experiments, we borrowed code from two separate, prior implementations of MAML: for sinusoid regression, we used the Tensorflow [code](https://github.com/cbfinn/maml) from the [original MAML paper](https://arxiv.org/pdf/1703.03400.pdf)), and for image classification, we used the Pytorch [code](https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch) from the ["How to train your MAML"](https://arxiv.org/pdf/1810.09502.pdf) paper. In both cases we implemented TR-MAML on top of the MAML implementation. Our code submission has two base directories - sinusoid/ for the sinusoid regression experiments and image_classification/ for the image classification experiments. 

## Sinusoid Regression

The contents of the "sinusoid" folder are for the sinusoid regression experiments. All requirements can be downloaded and installed by following the instructions in README.md in the original MAML repository. Depending on your version of Tensorflow, you may have to change "import tensorflow.compat.v1 as tf" to "import tensorflow as tf" in main.py and maml.py. 
The sinusoid regression experiments use synthetic data, so no dataset is required.
Our code runs exactly as the original MAML code runs with only 3 additional hyperparameters:
 - TR_MAML : boolean value indicating whether to use TR_MAML (True) or MAML (False) (default: False)
 - p_lr : float specifying the learning rate for p (default: 0.0001)
 
These hyperparameter are specified as flags in the bash command used to run the experiment. To run any experiment, navigate to the "sinusoid" directory and make sure all requirements are installed. Then issue one of the following commands:

### 5-shot sinusoid TR-MAML train:
python main.py --datasource=sinusoid --logdir=logs/sine/ --log_number=50 --p_lr=0.00001 --TR_MAML=True   --metatrain_iterations=70000 --norm=None --update_batch_size=5 --train=True --resume=False

### 5-shot sinusoid MAML train:
python main.py --datasource=sinusoid --logdir=logs/sine/ --log_number=60 --TR_MAML=False --metatrain_iterations=70000 --norm=None --update_batch_size=5 --train=True --resume=False

### 10-shot sinusoid TR-MAML train:
python main.py --datasource=sinusoid --logdir=logs/sine/ --log_number=100 --p_lr=0.00002 --TR_MAML=True   --metatrain_iterations=70000 --norm=None --update_batch_size=10 --train=True --resume=False

### 10-shot sinusoid MAML train:
python main.py --datasource=sinusoid --logdir=logs/sine/ --log_number=110 --TR_MAML=False   --metatrain_iterations=70000 --norm=None --update_batch_size=10 --train=True --resume=False


### 5-shot sinusoid TR-MAML test:
python main.py --datasource=sinusoid --logdir=logs/sine/ --log_number=50 --p_lr=0.00001 --TR_MAML=True   -metatrain_iterations=70000 --norm=None --update_batch_size=5 --train=False --resume=True

### 5-shot sinusoid MAML test:
python main.py --datasource=sinusoid --logdir=logs/sine/ --log_number=60 --TR_MAML=False --metatrain_iterations=70000 --norm=None --update_batch_size=5 --train=False --resume=True

### 10-shot sinusoid TR-MAML test:
python main.py --datasource=sinusoid --logdir=logs/sine/ --log_number=100 --p_lr=0.00002 --TR_MAML=True   --metatrain_iterations=70000 --norm=None --update_batch_size=10 --train=False --resume=True

### 10-shot sinusoid MAML test:
python main.py --datasource=sinusoid --logdir=logs/sine/ --log_number=110 --TR_MAML=False   --metatrain_iterations=70000 --norm=None --update_batch_size=10 --train=False --resume=True


## Image Classification

We use the Omniglot and Mini-ImageNet datasets for our experiments. See image_classification/README.md for download and installation instructions, as well as additional details on the directory structure. Our experiments are run in the same manner as those run for the existing MAML implementation from the "How to train your MAML paper", but again with a small number of additional hyperparameters. These are:

 - TR_MAML : booloean value indicating whether to use TR_MAML (True) or MAML (False) (default: False)
 - p_lr : float specifying the learning rate for p (default: 0.0001)

Here the bash command used to run an experiment is specified within a shell script that references a JSON file which specifies the hyperparameter values. The experiments can be run as follows:

1. Make sure all requirements are installed and the appropriate environment is created by following the Installation instructions at the ["How to train your MAML" Github site](https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch).
2. Navigate to image_classification/experiment_scripts
3. Run bash \[experiment_name.sh\] \[GPU_ID\]

As mentioned previously, each shell script calls a different JSON file, each of which are housed in the omniglot/experiment_config folder. There is one shell script and one configuration file for every experiment we run. An example JSON configuration file, for the 5-way, 1-shot TR-MAML experiment on Omniglot, is as follows:

{
  "batch_size":8,
  "image_height":28,
  "image_width":28,
  "image_channels":1,
  "gpu_to_use":0,
  "num_dataprovider_workers":4,
  "max_models_to_save":1,
  "dataset_name":"omniglot_dataset",
  "dataset_path":"omniglot_dataset",
  "reset_stored_paths":false,
  "experiment_name":"omni",
  "train_seed": 0, "val_seed": 0,
  "train_val_test_split": [0.70918052988, 0.03080714725, 0.2606284658],
  "indexes_of_folders_indicating_class": [-3, -2],
  "load_from_npz_files": false,
  "sets_are_pre_split": false,
  "load_into_memory": true,
  "init_inner_loop_learning_rate": 0.1,
  "train_in_stages": false,
  "multi_step_loss_num_epochs": 10,
  "minimum_per_task_contribution": 0.01,
  "num_evaluation_tasks":5000,
  "learnable_per_layer_per_step_inner_loop_learning_rate": false,
  "enable_inner_loop_optimizable_bn_params": false,

  "total_epochs": 12,
  "total_iter_per_epoch":5000, "continue_from_epoch": -2,
  "evaluate_on_test_set_only": false,
  "max_pooling": true,
  "per_step_bn_statistics": false,
  "learnable_batch_norm_momentum": false,
  "evalute_on_test_set_only": false,
  "learnable_bn_gamma": true,
  "learnable_bn_beta": true,

  "weight_decay": 0.0,
  "dropout_rate_value":0.0,
  "min_learning_rate":0.000001,
  "meta_learning_rate":0.001,   "total_epochs_before_pause": 100,
  "first_order_to_second_order_epoch":-1,
  "p_lr":0.00002,
  "TR_MAML":true,

  "norm_layer":"batch_norm",
  "cnn_num_filters":64,
  "num_stages":4,
  "conv_padding": true,
  "number_of_training_steps_per_iter":1,
  "number_of_evaluation_steps_per_iter":1,
  "cnn_blocks_per_stage":1,
  "num_classes_per_set":5,
  "num_samples_per_class":1,
  "num_target_samples": 10,

  "second_order": true,
  "use_multi_step_loss_optimization":false
}

Note: By downloading and using the mini-imagenet datasets, you accept terms and conditions found in imagenet_license.md



