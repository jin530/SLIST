# SLIST
This is officail code for the Web Conference 2021 paper: [`Session-aware Linear Item-Item Models for Session-based Recommendation`](https://arxiv.org/abs/2103.16104).</br>
We implemented our model based on the session-recommedndation framework [**session-rec**](https://github.com/rn5l/session-rec), and you can find the other session-based models and detailed usage on there.</br> 
Thank you for sharing the code.

**`README.md` and the comments in source code will be updated, again.**

The slides can be found [here](https://www.slideshare.net/ssuser1f2162/sessionaware-linear-itemitem-models-for-sessionbased-recommendation-www-2021).

## Dataset
Datasets can be downloaded from: </br>
https://www.dropbox.com/sh/n281js5mgsvao6s/AADQbYxSFVPCun5DfwtsSxeda?dl=0

- Unzip any dataset file to the data folder, i.e., rsc15-clicks.dat will then be in the folder data/rsc15/raw 
- Run a configuration with the following command:
For example: ```python run_preprocesing.py conf/preprocess/window/rsc15.yml```

## Basic Usage
- Change the expeimental settings and the model hyperparameters using a configuration file `*.yml`. </br>
- When a configuration file in conf/in has been executed, it will be moved to the folder conf/out.
- Run `run_config.py` with configuaration folder arguments to train and test models. </br>
For example: ```python run_confg.py conf/in conf/out```

## Running SLIST
- The yml files for slist used in paper can be found in `conf/save_slist`

## Requirements
- Python 3
- NumPy
- Pyyaml
- SciPy
- Sklearn
- Pandas
- Psutil

<!--
## Citation
Please cite our papaer:
```
@inproceedings{}
```
-->
