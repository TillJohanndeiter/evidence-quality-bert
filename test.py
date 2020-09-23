import time
import argparse
import pathlib

from random import shuffle
from tensorflow.keras.models import load_model

from preprocssing import ArgPair, load_arg_pairs

# For better testing purpose you can pass the path of the model and dataset which is used
parser = argparse.ArgumentParser()


parser.add_argument('--num',
                    default=5,
                    nargs='?',
                    help='Number of randomly chosen examples which will be predicted',
                    type=str)
parser.add_argument('model_filepath',
                    default='savedModel',
                    nargs='?',
                    help='Absolute or relative filepath of saved model file',
                    type=str)

parser.add_argument('dataset_filepath',
                    default='data',
                    nargs='?',
                    help='Absolute or relative filepath to folder with train.csv and test.csv',
                    type=str)

args = parser.parse_args()

model_filepath = args.model_filepath

model = load_model(model_filepath)

dataset_filepath = args.dataset_filepath

arg_pairs = load_arg_pairs(pathlib.Path(dataset_filepath).joinpath('test.csv'))

arg_counter = args.num


def print_argument(arg_pair : ArgPair):
    print('No you have ten seconds to think about which argument you '
          'would prefer in a discussion about {} '.format(arg_pair.topic))
    print('First argument has stance {} : ')
    print(arg_pair.first_arg)
    print('Second argument has stance {} : ')
    print(arg_pair.second_arg)
    time.sleep(10)


if __name__ == '__main__':

    print('Presentation starts: ')
    print('Here are {} randomly chosen examples from the dataset: '.format(arg_counter))
    arg_pairs = load_arg_pairs(dataset_filepath)
    arg_pairs = shuffle(arg_pairs)[:arg_counter]

    assert len(arg_pairs) == arg_counter
