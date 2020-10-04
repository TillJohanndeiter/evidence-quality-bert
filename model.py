import tensorflow_hub as hub
from numpy import int32

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling1D
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall,\
    TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from tensorflow.keras.optimizers import Adam

# Load layer from tfhub
BERT_MODEL_HUB = "https://tfhub.dev/tensorflow/bert_en_wwm_cased_L-24_H-1024_A-16/2"
BERT_LAYER = hub.KerasLayer(BERT_MODEL_HUB, trainable=True, name='bert_layer')

# Maximal length of two evidences in dataset.
MAX_SEQ_LENGTH = 125

METRICS = [BinaryAccuracy(),
           Precision(),
           Recall(),
           TruePositives(),
           TrueNegatives(),
           FalsePositives(),
           FalseNegatives()]


def simple_bert() -> Model:
    """
    Creates model using pretrained bert layer form google. Has three inputs
    (word_ids, masks and segment_ids). For explanation please look into
    documentation form preprocessing. Print summary after compilation.
    :return: created model
    """
    # TODO: Decide for model

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
    model.summary()

    return model
