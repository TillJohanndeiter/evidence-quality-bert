# BERT based evidence quality classificator

Pretrained transformer bert is used to solve the task of binary evidence comparsion based on a IBM dataset. The neuronal network classify the better evidence. The preprocessing, model and training process is implemented in a regular python skript and a jupyter notebook. Also a competive game agianst the neuronal network is includeded.

Supported python version: 3.8

## Installation

You could use a virtual environment to avoid dependency problems

```bash
python3 -m pip install --user virtualenv

python3 -m venv env

source env/bin/activate
```

Install requirements

```bash
pip install -r requirements.txt
```

To download the dataset please run.

```bash
bash download_dataset.sh
```

My trained model can be downloaded [here](https://drive.google.com/drive/folders/11TQd51OPjRIZmLkh1cvh_8-VmwPWkHjH?usp=sharing) (~1 GB) 

## Manual

**evi_bert.py**

Script preprocess dataset and and creates model. After this training and testing will be started. Then the trained model will be saved. At startup the Bert layer will be downloaded. This might take a while. 

Arguments:

dataset_filepath - Filepath to folder with train.csv and test.csv

savepath - Filepath of model safe folder

As default ./data and ./model_%Y%m%d-%H%M%S is used.

Example:

```bash
evi_bert.py ./data_path ./save_path
```

**evi_bert.ipynb**

evi_bert.py in notebook format. I recommend to use google colab.

**game.py**

Script will only work with saved model. You can use your trained instance or mine.
It will start a game in which the user and the model will predict the label of some pairs. Likewise in evi_bert.py the Bert layer will be downloaded at start. The layer is required for preprocessing only. This might take a while.

Arguments:

dataset_filepath - Path of test.csv

model_filepath - Filepath of saved model

--num - Number of randomly chosen pairs to predict

As default ./data/test.csv ./saved_model --num=5 is used.

Example:

```bash
game.py ./data_path/test.csv ./save_path --num=100
```

## License

[The Unlicense](https://choosealicense.com/licenses/unlicense/)


## Contact

If you have any feedback or an issue, please contact me at till.johanndeiter (at) web.de
