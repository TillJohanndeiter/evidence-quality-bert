# Argument Quality task: IBM Debater – Evidence Quality

Welcome to my natural language processing project.

## Installation

You could use a virtual environment to avoid dependency problems.  
Recommended if you use other tensorflow versions than 2.3.0. For this you have to create one and activate it:

```bash
python3 -m pip install --user virtualenv

python3 -m venv env

source env/bin/activate

```

Anyway install requirements from project folder

```bash
pip install -r requirements.txt
```

List of requirements:

numpy~=1.18.5
pandas~=1.1.2
tensorflow~=2.3.0
pip~=20.2.3
bert-for-tf2~=0.14.6
tensorflow_hub~=0.9.0


The dataset is available at [IBM research] (https://www.research.ibm.com/haifa/dept/vst/debating_data.shtml). Please download IBM Debater® - Evidence Quality dataset from section two *Argument Quality*.

My trained model can be downloaded [here](https://drive.google.com/drive/folders/11TQd51OPjRIZmLkh1cvh_8-VmwPWkHjH?usp=sharing) (~1 GB) 

## Manual

Project contains four python scripts.

**model.py**

Provides simple method evi_bert which creates model

**preprocessing.py**

Provides x_and_y_from_evi_pair which converts the dataset into keras compatible numpy arrays

**train-evaluate-main.py**

Script load dataset and creates model using model.py and preprocessing.py. After this training and testing will be started. After this model will be saved. At startup the Bert layer will be downloaded. This might take a while. 

Arguments:

dataset_filepath - Filepath to folder with train.csv and test.csv

savepath - Filepath of model safe folder

As default ./data and ./model_%Y%m%d-%H%M%S is used.

Example:

```bash
train-evaluate-main.py ./data_path ./save_path
```

**test.py**

Script will only work with saved model. You can use your trained instance or mine.
It will start a game in which the user and the model will predict the label of some pairs. Likewise in train-evaluate-main.py the Bert layer will be downloaded at start.
The layer ist required for preprocessing only. This might take a while.

Arguments:

dataset_filepath - Path of test.csv

model_filepath - Filepath of saved model

--num - Number of randomly chosen pairs which will be predicted

As default ./data/test.csv ./saved_model --num=5 is used.

Example:

```bash
test.py ./data_path/test.csv ./save_path --num=100
```

## License

[The Unlicense](https://choosealicense.com/licenses/unlicense/)


## Contact

If you have any feedback or an issue, please contact me at tjohanndeiter@techfak.de