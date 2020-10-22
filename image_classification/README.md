# Note: This readme is an editted version of the original readme from the "How to train your MAML" paper. All instructions here are consistent with the TR-MAML paper.

# How to train your MAML in Pytorch
A replication of the paper ["How to train your MAML"](https://arxiv.org/abs/1810.09502), along with a replication of the original ["Model Agnostic Meta Learning"](https://arxiv.org/abs/1703.03400) (MAML) paper.

By using this codebase you agree to the terms 
and conditions in the [LICENSE](https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch/blob/master/LICENSE) file.

## Installation

The code uses Pytorch to run, along with many other smaller packages. To take care of everything at once, we recommend 
using the conda package management library. More specifically, 
[miniconda3](https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh), as it is lightweight and fast to install.
If you have an existing miniconda3 installation please start at step 3. 
If you want to  install both conda and the required packages, please run:
 1. ```wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh```
 2. Go through the installation.
 3. Activate conda
 4. conda create -n meta_learning_pytorch_env python=3.6.
 5. conda activate meta_learning_pytorch_env
 6. At this stage you need to choose which version of pytorch you need by visiting [here](https://pytorch.org/get-started/locally/)
 7. Choose and install the pytorch variant of your choice using the conda commands.
 8. Then run ```bash install.sh```

To execute an installation script simply run:
```bash <installation_file_name>```

To activate your conda installations simply run:
```conda activate```

## Datasets

The Omniglot dataset can be downloaded from https://github.com/brendenlake/omniglot/tree/master/python, specifically images_background.zip and images_evaluation.zip must be downloaded. Once unzipped, these folders must be placed in  tr-maml/omniglot/datasets/omniglot_dataset, following the analogous structure as in the ["How to train your MAML" repository](https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch/).

The mini-ImageNet dataset can be downloaded from this Google Drive folder:
https://drive.google.com/file/d/1qQCoGoEJKUCQkk8roncWH7rhPN7aMfBr/view
All that needs to be done is place the downloaded .pbzip file in the /datasets directory, and the code will automate the rest.

Note: By downloading and using the mini-imagenet datasets, you accept terms and conditions found in imagenet_license.md

## Overview of code:

- datasets folder: Contains the dataset files (once added) and folders containing the images in a structure readable by the 
custom data provider.
- experiments_config: Contains configuration files for each and every experiment listed in the experiments script folder.
- experiment_scripts: Contains scripts that can reproduce every results in the paper. Each script is easily runnable
simply by executing:
```bash <experiment-script.sh> <GPU_ID>```
- script_generation_tools: Contains scripts and template files for the automatic generation of experiment scripts.
- utils: Contains utilities for dataset extraction, parser argument extraction and storage of statistics and others.
- data.py: Contains the data providers for the few shot meta learning task generation. 
- experiment_builder.py: Builds an experiment ready to train and evaluate your meta learning models. It supports automatic
checkpoining and even fault-tolerant code. If your script is killed for whatever reason, you can simply rerun the script.
It will find where it was before it was killed and continue onwards towards convergence.

- few_shot_learning_system.py: Contains the meta_learning_system class which is where most of MAML and TR-MAML are actually
implemented. It takes care of inner and outer loop optimization, checkpointing, reloading and statistics generation, as 
well as setting the rng seeds in pytorch.

- meta_neural_network_architectures: Contains new pytorch layers which are capable of utilizing either internal 
parameter or externally passed parameters.

- train_maml_system.py: A very minimal script that combines the data provider with a meta learning system and sends them
 to the experiment builder to run an experiment. Also takes care of automated extraction of data if they are not 
 available in a folder structure.

# Running an experiment

To run an experiment from the paper on Omniglot:
1. Activate your conda environment ```conda activate pytorch_meta_learning_env```
2. cd experiment_scripts
3. Find which experiment you want to run.
4. ```bash experiment_script.sh gpu_ids_separated_by_spaces```
