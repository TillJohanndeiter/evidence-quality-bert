"""
Start preprocessing, training and test evaluation.
The dataset 'IBM Debater® - Evidence Quality' is available at:
https://www.research.ibm.com/haifa/dept/vst/debating_data.shtml
The code, especially tokenization and formatting are inspired form official bert_model repository.
Source (Apache 2.0 license): https://github.com/google-research/bert/blob/master/run_classifier.py
Method convert_examples_to_features and associated help methods were customized to
the domain of dataset and unnecessary code parts are removed.
"""

import pathlib
import argparse
from datetime import datetime
from pathlib import Path

import numpy
from numpy import int32
import pandas

from bert import bert_tokenization

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling1D
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, \
    TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from tensorflow.keras.optimizers import Adam

import tensorflow_hub as hub

# Load layer from tfhub
BERT_MODEL_HUB = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
BERT_LAYER = hub.KerasLayer(BERT_MODEL_HUB, trainable=True, name='bert_layer')

# Maximal length of two evidences in dataset.
MAX_SEQ_LENGTH = 100

METRICS = [BinaryAccuracy(),
           Precision(),
           Recall(),
           TruePositives(),
           TrueNegatives(),
           FalsePositives(),
           FalseNegatives()]

# Vocabulary and lower case flag is loaded dynamic from layer.
vocab_file = BERT_LAYER.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = BERT_LAYER.resolved_object.do_lower_case.numpy()
tokenizer = bert_tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


def evi_bert() -> Model:
    """
    Creates model using pretrained bert_model layer form google. Has three inputs
    (word_ids, masks and segment_ids). For explanation please look into
    documentation form preprocessing. Print summary after compilation.
    :return: created model
    """

    input_word_ids = Input(shape=(MAX_SEQ_LENGTH,), dtype=int32,
                           name="input_word_ids")
    input_mask = Input(shape=(MAX_SEQ_LENGTH,), dtype=int32,
                       name="input_mask")
    segment_ids = Input(shape=(MAX_SEQ_LENGTH,), dtype=int32,
                        name="segment_ids")
    bert_inputs = [input_word_ids, input_mask, segment_ids]

    _, sequence_output = BERT_LAYER(bert_inputs)
    sequence_output = GlobalAveragePooling1D(name='average_over_time')(sequence_output)

    output = Dense(1, activation='sigmoid')(sequence_output)

    model = Model(inputs=bert_inputs, outputs=output, name='evi_bert')
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-5),
                  metrics=METRICS)

    return model


class EviPair():
    """A single pair of two evidence which are ranked by the label."""

    def __init__(self,
                 first_evi: str,
                 second_evi: str,
                 first_stance: str,
                 second_stance: str,
                 topic: str,
                 acceptance_rate: str,
                 label: int, ):
        """
        evis:
          first_evi: string. The untokenize text of the first evidence from the dataset.
          second_evi: string. The untokenize text of the second evidence from the dataset.
          label: string. The label of the data sample. Label equals one means first one is
          the better evidence. Label equals second means second evidence is better.
        """
        self.first_evi = first_evi
        self.second_evi = second_evi
        self.first_stance = first_stance
        self.second_stance = second_stance
        self.topic = topic
        self.acceptance_rate = acceptance_rate
        self.label = label


class ProcessedEviPair():
    """Representation of tokenized and process evidence pair. Contains arrays of corresponding
  token_ids, which makes an sentence understandable by bert_model. Input_mask shows which part of
  token_ids are padded. Segment ids signalize which part of token_ids belong to first and second
  evidence."""

    def __init__(self,
                 token_ids,
                 input_mask,
                 segment_ids,
                 label_id, ):
        self.token_ids = token_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label_id


def _truncate_evi_pair(tokens_a: [str], tokens_b: [str]):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= MAX_SEQ_LENGTH - 3:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def _process_evi_pair(evi_pair: EviPair) -> ProcessedEviPair:
    """
    The convention in BERT is:
    For sequence pairs:
    tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    input_ids: Corresponding numerical ids for each token.
    input_mask : mask has 1 for real tokens and 0 for padding tokens
    segment_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    Segment_ids are used to indicate whether this is the first
    sequence or the second sequence.
    Input_ids are used as sentence representation.
    :param evi_pair: currently processed evidence pair
    :return: ProcessedEviPair of evi_pair
    """

    first_evi_tokens = tokenizer.tokenize(evi_pair.first_evi)
    second_evi_tokens = tokenizer.tokenize(evi_pair.second_evi)

    _truncate_evi_pair(first_evi_tokens, second_evi_tokens)

    segment_ids, tokens = create_tokens(first_evi_tokens, second_evi_tokens)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # 1 is appended len of input_ids to mark non padded part of input
    input_mask = [1] * len(input_ids)

    _pad_up_to_max_seq_len(input_ids, input_mask, segment_ids)

    assert len(input_ids) == MAX_SEQ_LENGTH
    assert len(input_mask) == MAX_SEQ_LENGTH
    assert len(segment_ids) == MAX_SEQ_LENGTH

    # Binary label is corrected form [1, 2] to [0,1]
    return ProcessedEviPair(
        token_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=evi_pair.label - 1)


def create_tokens(first_evi_tokens, sec_evi_tokens) -> ([], []):
    """
    Create segments_ids (0 for first evidence, 1 for second evidence)
    and add [SEP] and [CLS] and merge first_evi_tokens and sec_evi_tokens
    :param first_evi_tokens: tokens of first evidence
    :param sec_evi_tokens: second of first evidence
    :return: segment_ids and tokens for both evidences
    """
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in first_evi_tokens:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    if sec_evi_tokens:
        for token in sec_evi_tokens:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
    return segment_ids, tokens


def _pad_up_to_max_seq_len(input_ids, input_mask, segment_ids):
    """
    Add zeros to input_ids
    :param input_ids: input_ids for currently processed evidence
    :param input_mask: input_mask for currently processed evidence
    :param segment_ids: segment_ids for currently processed evidence
    :return: None
    """
    while len(input_ids) < MAX_SEQ_LENGTH:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)


def _process_evi_pairs(evi_pairs: []):
    """Convert a set of eviPairs to a list of ProcessedEviPair."""
    return [_process_evi_pair(example) for example in evi_pairs]


def _convert_to_numpy_arrays(evidences: [ProcessedEviPair]) \
        -> ((numpy.ndarray, numpy.ndarray, numpy.ndarray), numpy.ndarray):
    """
    Convert list of ProcessedEviPair's to 4 numpy arrays which are compatible with
    keras and tensorflow
    :param evidences: list of ProcessedEviPair's to convert
    :return: tuples of one and three numpy arrays (token_ids, attention_mask, type_ids) labels
    """
    return (numpy.stack([evi.token_ids for evi in evidences]),
            numpy.stack([evi.input_mask for evi in evidences], ),
            numpy.stack([evi.segment_ids for evi in evidences], )), \
           numpy.stack([evi.label for evi in evidences])


def x_and_y_from_dataset(filepath: Path) -> \
        ((numpy.ndarray, numpy.ndarray, numpy.ndarray), numpy.ndarray):
    """
    Main function which receives absolute or relative filepath of an csv file
    which will be tokenized, type/attention masks and labels are created. Furthermore
    a tuple of numpy arrays is created which is directly usable for keras.
    :param filepath: absolute or relative path of csv file
    :return: tuples of one and three numpy arrays (token_ids, attention_mask, type_ids) labels
    """

    # Create instance of EviPair with correct label for
    evi_pairs = load_evi_pairs(filepath)
    processed_evi_pairs = _process_evi_pairs(evi_pairs)
    x_numpy, y_numpy = _convert_to_numpy_arrays(processed_evi_pairs)

    # Check that entry and y have same length
    assert len(x_numpy[0]) == len(x_numpy[1]) == len(x_numpy[2]) == len(y_numpy)

    # Check that all input vectors have same length as MAX_SEQ_LENGTH
    assert x_numpy[0].shape[1] == x_numpy[1].shape[1] == x_numpy[2].shape[1] == MAX_SEQ_LENGTH

    return x_numpy, y_numpy


def x_and_y_from_evi_pair(evi_pair: EviPair) -> \
        ((numpy.ndarray, numpy.ndarray, numpy.ndarray), numpy.ndarray):
    """
    Convert evi_pair to keras usable numpy arrays
    :param evi_pair: Sample from dataset which will be converted
    :return: numpy arrays which can be used with keras
    """
    process_evi_pair = _process_evi_pair(evi_pair)
    return _convert_to_numpy_arrays([process_evi_pair])


def load_evi_pairs(filepath) -> [EviPair]:
    """
    Load csv dataset from filepath and convert it to dataframe and
    afterwards to list of EviPair's.
    :param filepath: filepath of csv dataset to parse
    :return: list of EviPair's
    """
    dataframe = pandas.read_csv(filepath)
    evi_pairs = dataframe.apply(lambda entry: EviPair(first_evi=entry[1],
                                                      second_evi=entry[2],
                                                      topic=entry[0],
                                                      first_stance=entry[5],
                                                      second_stance=entry[6],
                                                      acceptance_rate=entry[4],
                                                      label=entry[3]), axis=1)
    return evi_pairs.tolist()


# For better testing purpose you can pass the path of the folder of test and train csv
parser = argparse.ArgumentParser()
parser.add_argument('filepath',
                    default='data',
                    nargs='?',
                    help='Filepath to folder with train.csv and test.csv',
                    type=str)

parser.add_argument('dataset_filepath',
                    default='model_{}'.format(datetime.now().strftime("%Y%m%d-%H%M%S")),
                    nargs='?',
                    help='Filepath of model safe folder',
                    type=str)

"""
Load test and train set. Afterwards received model and start training
season. Training will generate a file for tensorboard.
After that evaluation and print of metrics is done. At the end
model is saved.
"""
if __name__ == '__main__':

    args = parser.parse_args()

    data_path = pathlib.Path(args.filepath)

    dataset_filepath = args.dataset_filepath

    train_path = data_path.joinpath('train.csv')
    test_path = data_path.joinpath('train.csv')

    if train_path.is_file() and test_path.is_file():

        train_x, train_y = x_and_y_from_dataset(train_path)
        test_x, test_y = x_and_y_from_dataset(test_path)

        # Correct length of tuple
        assert len(train_x) == len(test_x) == 3

        bert_model = evi_bert()
        bert_model.summary()

        assert bert_model is not None

        bert_model.fit(x=train_x, y=train_y,
                       batch_size=16, epochs=10,
                       callbacks=[EarlyStopping()],
                       validation_split=0.2, shuffle=True)

        scores = bert_model.evaluate(x=test_x, y=test_y, verbose=1)

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

        bert_model.save(dataset_filepath)

    else:
        print('Datasets at  {} and {} not found!'.format(train_path, test_path))
