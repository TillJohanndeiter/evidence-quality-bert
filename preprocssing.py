import numpy
import pandas
from bert import bert_tokenization
import tensorflow_hub as hub

MAX_SEQ_LENGTH = 200  # Maximal length of two argument in dataset.
LABLES = [0, 1]
lengths = set()

# Load layer from tfhub
BERT_MODEL_HUB = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"
BERT_LAYER = hub.KerasLayer(BERT_MODEL_HUB, trainable=True)

# Vocabulary and lower case flag is loaded dynamic from layer.
vocab_file = BERT_LAYER.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = BERT_LAYER.resolved_object.do_lower_case.numpy()
tokenizer = bert_tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


class ArgPair(object):
    """A single pair of two arguments which are ranked by the label."""

    def __init__(self,
                 first_arg,
                 second_arg,
                 label):
        """Constructs a ArgPair.
        Args:
          first_arg: string. The untokenize text of the first argument from the dataset.
          second_arg: string. The untokenize text of the second argument from the dataset.
          label: string. The label of the data sample. Label equals one means first one is
          the better argument. Label equals second means second argument is better.
        """
        self.first_arg = first_arg
        self.second_arg = second_arg
        self.label = label


class ProcessedArgPair(object):
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
        lengths.add(total_length)
        assert (total_length <= MAX_SEQ_LENGTH - 3)
        if total_length <= MAX_SEQ_LENGTH - 3:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def _process_arg_pair(argPair: ArgPair) -> ProcessedArgPair:
    # The mask has 1 for real tokens and 0 for padding tokens.
    first_arg_tokens = tokenizer.tokenize(argPair.first_arg)
    second_arg_tokens = tokenizer.tokenize(argPair.second_arg)

    # TODO: Remove
    # _truncate_arg_pair(first_arg_tokens, second_arg_tokens)

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
        label_id=argPair.label)


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


def create_X_and_Y(filepath: str) -> ((numpy.ndarray, numpy.ndarray, numpy.ndarray), numpy.ndarray):
    """
    Main function which receives absolute or relative filepath of an csv file
    which will be tokenized, type/attention masks and labels are created. Furthermore
    a tuple of numpy arrays is created which is directly usable for keras.
    :param filepath: absolute or relative path of
    :return: tuples of one and three numpy arrays (token_ids, attention_mask, type_ids) labels
    """
    dataset = pandas.read_csv(filepath)

    # Create instance of ArgPair with correct label for
    arg_pairs = dataset.apply(lambda entry: ArgPair(first_arg=entry[1],
                                                    second_arg=entry[2],
                                                    label=entry[3] - 1), axis=1)
    processed_arg_pairs = _process_arg_pairs(arg_pairs)
    x, y = _convert_to_numpy_arrays(processed_arg_pairs)

    # Check that entry and y have same length
    assert len(x[0]) == len(x[1]) == len(x[2]) == len(y)

    # Check that all input vectors have same length as MAX_SEQ_LENGTH
    assert x[0].shape[1] == x[1].shape[1] == x[2].shape[1] == MAX_SEQ_LENGTH

    return x, y


if __name__ == '__main__':
    x, y = create_X_and_Y('data/train.csv')
