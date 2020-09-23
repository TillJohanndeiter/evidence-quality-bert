"""
Module provides main method create_x_and_y which creates keras compatible numpy arrays
from a csv dataset. All other methods are help methods create argument pairs, tokenize
arguments and create the bert compatible format, which consists of word, attention and
type ids. The dataset 'IBM Debater® - Evidence Quality' is available at:
https://www.research.ibm.com/haifa/dept/vst/debating_data.shtml
The code, especially tokenization and formatting are inspired form official bert repository.
Source (Apache 2.0 license): https://github.com/google-research/bert/blob/master/run_classifier.py
Method convert_examples_to_features and associated help methods were customized to
the domain of dataset and unnecessary code parts are removed.
"""

import numpy
import pandas

from bert import bert_tokenization
from model import BERT_LAYER, MAX_SEQ_LENGTH

# Vocabulary and lower case flag is loaded dynamic from layer.
vocab_file = BERT_LAYER.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = BERT_LAYER.resolved_object.do_lower_case.numpy()
tokenizer = bert_tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


class ArgPair():
    """A single pair of two arguments which are ranked by the label."""

    def __init__(self,
                 first_arg: str,
                 second_arg: str,
                 first_stance: str,
                 second_stance: str,
                 topic: str,
                 label: int, ):
        """
        Args:
          first_arg: string. The untokenize text of the first argument from the dataset.
          second_arg: string. The untokenize text of the second argument from the dataset.
          label: string. The label of the data sample. Label equals one means first one is
          the better argument. Label equals second means second argument is better.
        """
        self.first_arg = first_arg
        self.second_arg = second_arg
        self.first_stance = first_stance
        self.second_stance = second_stance
        self.topic = topic
        self.label = label


class ProcessedArgPair():
    """Representation of tokenized and process argument pair. Contains arrays of corresponding
  token_ids, which makes an sentence understandable by bert. Input_mask shows which part of
  token_ids are padded. Segment ids signalize which part of token_ids belong to first and second
  argument."""

    def __init__(self,
                 token_ids,
                 input_mask,
                 segment_ids,
                 label_id, ):
        self.token_ids = token_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label_id


def _truncate_arg_pair(tokens_a, tokens_b):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    # TODO: Decide if truncating is needed
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= MAX_SEQ_LENGTH - 3:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def _process_arg_pair(arg_pair: ArgPair) -> ProcessedArgPair:
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
    :param arg_pair: currently processed argument pair
    :return: ProcessedArgPair of arg_pair
    """

    first_arg_tokens = tokenizer.tokenize(arg_pair.first_arg)
    second_arg_tokens = tokenizer.tokenize(arg_pair.second_arg)

    # TODO: Remove
    _truncate_arg_pair(first_arg_tokens, second_arg_tokens)

    segment_ids, tokens = create_tokens(first_arg_tokens, second_arg_tokens)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # 1 is appended len of input_ids to mark non padded part of input
    input_mask = [1] * len(input_ids)

    _pad_up_to_max_seq_len(input_ids, input_mask, segment_ids)

    assert len(input_ids) == MAX_SEQ_LENGTH
    assert len(input_mask) == MAX_SEQ_LENGTH
    assert len(segment_ids) == MAX_SEQ_LENGTH

    return ProcessedArgPair(
        token_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=arg_pair.label)


def create_tokens(first_arg_tokens, sec_arg_tokens) -> ([], []):
    """
    Create segments_ids (0 for first argument, 1 for second argument)
    and add [SEP] and [CLS] and merge first_arg_tokens and sec_arg_tokens
    :param first_arg_tokens: tokens of first argument
    :param sec_arg_tokens: second of first argument
    :return: segment_ids and tokens for both arguments
    """
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in first_arg_tokens:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    if sec_arg_tokens:
        for token in sec_arg_tokens:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
    return segment_ids, tokens


def _pad_up_to_max_seq_len(input_ids, input_mask, segment_ids):
    """
    Add zeros to input_ids
    :param input_ids: input_ids for currently processed word
    :param input_mask: input_mask for currently processed word
    :param segment_ids: segment_ids for currently processed word
    :return: None
    """
    while len(input_ids) < MAX_SEQ_LENGTH:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)


def _process_arg_pairs(examples):
    """Convert a set of ArgPairs to a list of ProcessedArgPair."""
    return [_process_arg_pair(example) for example in examples]


def _convert_to_numpy_arrays(arguments: [ProcessedArgPair]) \
        -> ((numpy.ndarray, numpy.ndarray, numpy.ndarray), numpy.ndarray):
    """
    Convert list of ProcessedArgPairs to 4 numpy arrays which are compatible with
    keras and tensorflow
    :param arguments: list of ProcessedArgPairs to convert
    :return: tuples of one and three numpy arrays (token_ids, attention_mask, type_ids) labels
    """
    return (numpy.stack([arg.token_ids for arg in arguments]),
            numpy.stack([arg.input_mask for arg in arguments], ),
            numpy.stack([arg.segment_ids for arg in arguments], )), \
           numpy.stack([arg.label for arg in arguments])


def create_x_and_y(filepath: str) -> ((numpy.ndarray, numpy.ndarray, numpy.ndarray), numpy.ndarray):
    """
    Main function which receives absolute or relative filepath of an csv file
    which will be tokenized, type/attention masks and labels are created. Furthermore
    a tuple of numpy arrays is created which is directly usable for keras.
    :param filepath: absolute or relative path of
    :return: tuples of one and three numpy arrays (token_ids, attention_mask, type_ids) labels
    """

    # Create instance of ArgPair with correct label for
    arg_pairs = load_arg_pairs(filepath)
    processed_arg_pairs = _process_arg_pairs(arg_pairs)
    x_numpy, y_numpy = _convert_to_numpy_arrays(processed_arg_pairs)

    # Check that entry and y have same length
    assert len(x_numpy[0]) == len(x_numpy[1]) == len(x_numpy[2]) == len(y_numpy)

    # Check that all input vectors have same length as MAX_SEQ_LENGTH
    assert x_numpy[0].shape[1] == x_numpy[1].shape[1] == x_numpy[2].shape[1] == MAX_SEQ_LENGTH

    return x_numpy, y_numpy


def load_arg_pairs(filepath) -> [ArgPair]:
    """
    Load csv dataset from filepath and convert it to dataframe and
    afterwards to list of ArgPair's.
    :param filepath: filepath of csv dataset to parse
    :return: list of ArgPair's
    """
    dataframe = pandas.read_csv(filepath)
    arg_pairs = dataframe.apply(lambda entry: ArgPair(first_arg=entry[1],
                                                    second_arg=entry[2],
                                                    topic=entry[0],
                                                    first_stance=entry[5],
                                                    second_stance=entry[6],
                                                    label=entry[3] - 1), axis=1)
    return arg_pairs
