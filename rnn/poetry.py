import tensorflow as tf  
import numpy as np 
import collections

file = '../data/poetry/poetry.txt'


# 数据预处理

poetrys = []
with open(file) as f:
    for line in f:
        poetrys.append(line)

all_tolens = []
for poetry in poetrys:
    all_tolens += [word for word in poetry]

counter = collections.Counter(all_tolens)
words, _ = zip(*counter.most_common())

word_num_map = dict(zip(words, range(len(words))))
to_num = lambda word: word_num_map.get(word, len(words))
poetrys_vector = [list(map(to_num, poetry)) for poetry in poetrys]
x_array = []
y_array = []
for poetry_vector in poetrys_vector:
    x_array.append(np.array(poetry_vector[0:len(poetry_vector) - 1]))
    y_array.append(np.array(poetry_vector[1:len(poetry_vector)]))

# x_array = np.array(x_array)
# y_array = np.array(y_array)

# print(x_array.shape)
# print(y_array.shape)


# RNN 
class PoetryModel():
    def __init__(self, max_len, loss, opt, batch_size, epochs, vocab_size):
        self.max_len = max_len
        self.loss = loss 
        self.opt = opt
        self.batch_size = batch_size
        self.epochs = epochs
        self.vocab_size = vocab_size

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size + 1, 300, batch_input_shape=(self.batch_size, None)),
            tf.keras.layers.GRU(128, return_sequences=True, stateful=True),
            tf.keras.layers.Dense(self.vocab_size)
        ])
        model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
        return model

    def train(self, x_train, y_train, save_path, logpath):
        callbacks = [
            tf.keras.callbacks.TensorBoard(logpath),
            tf.keras.callbacks.ModelCheckpoint(save_path, save_best_only=True)
        ]
        self.model = self.build_model()
        self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, callbacks=callbacks)


if __name__ == "__main__":
    print(np.array(x_array))
    model = PoetryModel(128, 'sparse_categorical_crossentropy', 'adam', 100, 1, len(words))
    model.train(x_array, y_array, '../data/poetry', '../data/poetry')
    





