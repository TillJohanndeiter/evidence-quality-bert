"""
Module contains simple script which uses generate_x_and_y and simple_bert
to load data and get a model. After that training and evaluation is done.
"""

import pathlib
import argparse
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

from preprocssing import create_x_and_y
from model import simple_bert

# For better testing purpose you can pass the path of the folder of test and train csv
parser = argparse.ArgumentParser()
parser.add_argument('filepath',
                    default='data',
                    nargs='?',
                    help='Absolute or relative filepath to folder with train.csv and test.csv',
                    type=str)

args = parser.parse_args()

path = args.filepath

"""
Load test and train set. Afterwards received model and start training
season. After that evaluation and print of metrics is done.
"""

test_x, test_y = create_x_and_y(pathlib.Path(path).joinpath('test.csv'))
train_x, train_y = create_x_and_y(pathlib.Path(path).joinpath('test.csv'))

# Correct length of tuple
assert len(train_x) == len(test_x) == 3

simple_bert = simple_bert()

assert simple_bert is not None

simple_bert.fit(x=train_x, y=train_y,
                batch_size=128, epochs=1,
                callbacks=[TensorBoard(), EarlyStopping()],
                validation_split=0.2, shuffle=True)

scores = simple_bert.evaluate(x=test_x, y=test_y, verbose=1)

assert len(scores) > 6

# Print metrics

print('Binary Accuracy: {}'.format(scores[1]))
print('Precision: {}'.format(scores[2]))
print('Recall: {}'.format(scores[3]))

# Print values of confusion matrix

print('True negatives / Labeled with first argument '
      'and predicted as first argument  : {}'.format(scores[5]))
print('True positives / Labeled with second argument '
      'and predicted as second argument : {}'.format(scores[4]))
print('False negatives / Labeled with first argument '
      'and predicted as second argument : {}'.format(scores[7]))
print('False positives / Labeled with second argument '
      'and predicted as first argument : {}'.format(scores[6]))
