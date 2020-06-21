
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf 
import numpy as np 
import tensorflow_hub as hub 
import tensorflow_datasets as tfds

try:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

train_valication_split = tfds.Split.TRAIN.subsplit([6, 4])

(train_date, validation_data), test_data = tfds.load(name='imdb_reviews', split=(train_valication_split, tfds.Split.TEST), as_supervised=True)






