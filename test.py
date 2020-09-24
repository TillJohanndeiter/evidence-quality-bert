import argparse
import pathlib
import tensorflow as tf

assert tf.__version__ == '2.3.0'

from random import shuffle

from pip._vendor.distlib.compat import raw_input
print('Load layer. This might take a while.')
from tensorflow.keras.models import load_model

from preprocssing import EviPair, load_evi_pairs, x_and_y_from_evi_pair

# For better testing purpose you can pass the path of the model and dataset which is used
parser = argparse.ArgumentParser()

parser.add_argument('--num',
                    default=5,
                    nargs='?',
                    help='Number of randomly chosen arg_pairs which will be predicted',
                    type=int)
parser.add_argument('model_filepath',
                    default='saved_model',
                    nargs='?',
                    help='Absolute or relative filepath of saved model file',
                    type=str)

parser.add_argument('dataset_filepath',
                    default='data/test.csv',
                    nargs='?',
                    help='Absolute or relative filepath to folder with train.csv and test.csv',
                    type=str)

WRONG_INPUT = 'Please enter only 1 or 2'

args = parser.parse_args()

model_filepath = args.model_filepath

model = load_model(model_filepath)

dataset_filepath = args.dataset_filepath

arg_pairs = load_evi_pairs(pathlib.Path(dataset_filepath))

num_examples = args.num

human_correct_pred = 0
nn_correct_pred = 0
same_prediction = 0


def print_sample(arg_pair: EviPair):
    """
    Print evidences, stances, label and predication for one sample.
    :param arg_pair: Current EviPair
    :return: None
    """
    # TODO: print ID corresponding wikipedia article ?
    print('Please think about which argument you '
          'would prefer in a discussion about: '
          '\'{}\' '.format(arg_pair.topic))
    print('First evidence has stance {} : '.format(arg_pair.first_stance))
    print(arg_pair.first_evi)
    print('Second evidence has stance {} : '.format(arg_pair.second_stance))
    print(arg_pair.second_evi)
    print('Enter your choice: ')
    nn_prediction = predict_and_eval(arg_pair)
    print('Neuronal Network selected evidence {}'.format(nn_prediction))
    print('By an acceptance rate of {} sample was labeled as {} \n'.
          format(arg_pair.acceptance_rate, arg_pair.label))


def predict_and_eval(arg_pair: EviPair):
    """
    Make prediction by network and receive prediction from user.
    Afterwards will increase counter for correct predictions and
    amount of same predictions.
    :param arg_pair: Current sample
    :return: predicted label from network
    """
    global nn_correct_pred, same_prediction

    user_choice = get_user_input(arg_pair)
    x_input, _ = x_and_y_from_evi_pair(arg_pair)
    nn_prediction = model.predict(x_input)[0][0]

    # Mapping from probabilities to

    pred_class = 2 if nn_prediction > 0.5 else 1
    if pred_class == arg_pair.label:
        nn_correct_pred += 1
    if user_choice == pred_class:
        same_prediction += 1

    return pred_class


def get_user_input(arg_pair: EviPair):
    """
    Check if user entered a valid label in [1, 2].
    Otherwise will print message and repeat input.
    :param arg_pair: Current sample
    :return: choice of user as integer
    """
    global human_correct_pred

    while True:
        try:
            choice = int(raw_input())

            if choice in [1,2]:

                if choice == arg_pair.label:
                    human_correct_pred += 1

                break
            else:
                print(WRONG_INPUT)
        except ValueError:
            print(WRONG_INPUT)

    return choice


if __name__ == '__main__':

    print('Load Model successfully')
    model.summary()

    print('Demonstration starts: ')
    if num_examples == 1:
        print('Now one sample from the dataset with two evidences is shown to you.'
              ' The Evidence can be of same or opponent stance. '
              'So think of which you would take in an argument or '
              'which is more convincing.')
        print('Please evaluate the better evidence with 0 for first'
              ' and 1 for second evidence')
    else:
        print('Now {} samples from the dataset with two evidences are shown to you.'
              ' Evidences can be of same or opponent stance. '
              'So think of which you would take in an argument or '
              'which is more convincing.'.format(num_examples))
        print('Please evaluate the better evidences by enter 0 for first'
              ' and 1 for second evidence')

    # Load dataset, shuffle and slice wanted number of examples

    arg_pairs = load_evi_pairs(dataset_filepath)
    shuffle(arg_pairs)

    arg_pairs = arg_pairs[:num_examples]

    # Repeat prediction for each sample
    assert len(arg_pairs) == num_examples

    for arg_pair in arg_pairs:
        print_sample(arg_pair)

    # Print scores and "funny" message

    print('Your score: {} / {}'.format(human_correct_pred, num_examples))
    print('Score of our Network: {} / {}'.format(nn_correct_pred, num_examples))
    print('You and our computational intelligence selected the '
          'same evidence of {} samples'.format(same_prediction))

    if human_correct_pred == nn_correct_pred:
        print('You and our network have the same accuracy')
    elif human_correct_pred > nn_correct_pred:
        print('You are a better evidence detector than our network!')
    else:
        print('Even the computer is better than you. You have to '
              'train your evidence detector evaluation skills!')

    print('Well done! I hope you enjoyed the demonstration')
