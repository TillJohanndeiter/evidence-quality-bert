"""
Module contains simple script which uses generate_x_and_y and simple_bert
to load data and get a model. After that training and evaluation is done.
"""

import pathlib
import argparse
from datetime import datetime

import tensorflow as tf

assert tf.__version__ == '2.3.0'

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

from preprocssing import x_and_y_from_dataset
from model import simple_bert

# For better testing purpose you can pass the path of the folder of test and train csv
parser = argparse.ArgumentParser()
parser.add_argument('filepath',
                    default='data',
                    nargs='?',
                    help='Absolute or relative filepath to folder with train.csv and test.csv',
                    type=str)

parser.add_argument('savepath',
                    default='model {}'.format(datetime.now().strftime("%Y%m%d-%H%M%S")),
                    nargs='?',
                    help='Absolute or relative filepath the save folder',
                    type=str)

args = parser.parse_args()

data_path = args.filepath

save_path = args.savepath
"""
Load test and train set. Afterwards received model and start training
season. Training will generate a file for tensorboard.
After that evaluation and print of metrics is done. At the end
model is saved.
"""
if __name__ == '__main__':
    test_x, test_y = x_and_y_from_dataset(pathlib.Path(data_path).joinpath('test.csv'))
    train_x, train_y = x_and_y_from_dataset(pathlib.Path(data_path).joinpath('test.csv'))

    # Correct length of tuple
    assert len(train_x) == len(test_x) == 3

    simple_bert = simple_bert()

    assert simple_bert is not None

    simple_bert.fit(x=train_x, y=train_y,
                    batch_size=32, epochs=1,
                    callbacks=[TensorBoard(), EarlyStopping()],
                    validation_split=0.2, shuffle=True)

    scores = simple_bert.evaluate(x=test_x, y=test_y, verbose=1)

    assert len(scores) > 6

    # Print metrics

    print('Binary Accuracy: {}'.format(scores[1]))
    print('Precision: {}'.format(scores[2]))
    print('Recall: {}'.format(scores[3]))

    # Print values of confusion matrix

    print('True negatives / Labeled with first evidence '
          'and predicted as first evidence  : {}'.format(scores[5]))
    print('True positives / Labeled with second evidence '
          'and predicted as second evidence : {}'.format(scores[4]))
    print('False negatives / Labeled with first evidence '
          'and predicted as second evidence : {}'.format(scores[7]))
    print('False positives / Labeled with second evidence '
          'and predicted as first evidence : {}'.format(scores[6]))

    simple_bert.save(save_path)
