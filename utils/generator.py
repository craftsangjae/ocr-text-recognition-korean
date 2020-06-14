from tensorflow.keras.utils import Sequence
import numpy as np
import pandas as pd
from utils.dataset import OCRDataset


class OCRGenerator(Sequence):
    "Generates OCR TEXT Recognition Dataset for Keras"

    def __init__(self, dataset, char_list=None,
                 batch_size=32, blank_value=-1, shuffle=True):
        """
        Initialization

        param
        :param dataset : instance of class 'OCRDataset'
        :param char_list : unique character list (for Embedding)
        :param batch_size : the number of batch
        :param blank_value : the value of `blank` label
        :param shuffle : whether shuffle dataset or not
        """
        self.dataset = dataset
        if char_list is None:
            self.char_list = [chr(idx) for idx in range(ord('가'),ord('힣')+1)]
            self.char2idx = { char : idx for idx, char in enumerate(self.char_list)}
        else:
            self.char_list = char_list
            self.char2idx = {char: idx
                             for idx, char
                             in enumerate(self.char_list)}

        self.batch_size = batch_size
        self.max_length = self.dataset.max_word + 1 # With blank time step for Last label
        self.blank_value = blank_value
        self.shuffle = shuffle
        self.num_classes = len(self.char_list) + 1 # With Blank Token
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return len(self.dataset) // self.batch_size

    def __getitem__(self, index):
        "Generator one batch of dataset"
        images, texts = self.dataset[self.batch_size * index:
                                     self.batch_size * (index + 1)]
        # label sequence
        labels = np.ones([self.batch_size, self.max_length], np.int32)
        labels *= -1  # BLANK Token value : -1
        for idx, text in enumerate(texts):
            labels[idx, :len(text)] = np.array([self.char2idx[char] for char in text])
        return images, labels

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        if self.shuffle:
            self.dataset.shuffle()


class OCRSeq2SeqGenerator(Sequence):
    "Generates OCR TEXT Recognition Dataset for Keras"

    def __init__(self, dataset, char_list=None,
                 batch_size=32, blank_value=-1, shuffle=True,
                 return_initial_state=True, state_size=512):
        """
        Initialization

        param
        :param dataset : instance of class 'OCRDataset'
        :param char_list : unique character list (for Embedding)
        :param batch_size : the number of batch
        :param blank_value : the value of `blank` label
        :param shuffle : whether shuffle dataset or not
        :param return_initial_state : Whether return Initial state(Zero state) or not
        :param state_size : if return_initial_state is True, the size of initial state
        """
        self.dataset = dataset
        if char_list is None:
            self.char_list = [chr(idx) for idx in range(ord('가'),ord('힣')+1)]
            self.char2idx = { char : idx for idx, char in enumerate(self.char_list)}
        else:
            self.char_list = char_list
            self.char2idx = {char: idx
                             for idx, char
                             in enumerate(self.char_list)}

        self.batch_size = batch_size
        self.max_length = self.dataset.max_word + 1 # With <EOS> time step for Last label
        self.blank_value = blank_value
        self.shuffle = shuffle
        self.num_classes = len(self.char_list) + 1 # With <EOS> Token
        self.return_initial_state = return_initial_state
        self.state_size = state_size
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return len(self.dataset) // self.batch_size

    def __getitem__(self, index):
        "Generator one batch of dataset"
        images, texts = self.dataset[self.batch_size * index:
                                     self.batch_size * (index + 1)]
        # label sequence
        labels = np.ones([self.batch_size, self.max_length], np.int32)
        labels *= -1  # BLANK Token value : -1
        for idx, text in enumerate(texts):
            labels[idx, :len(text)] = np.array([self.char2idx[char] for char in text])
            labels[idx, len(text)] = self.num_classes # <EOS> Token

        target_inputs = np.roll(labels, 1, axis=1)
        target_inputs[:, 0] = self.num_classes # <EOS> Token
        target_inputs[target_inputs==-1] = self.num_classes # <EOS> Token

        X = {
            "images" : images,
            "decoder_inputs" : target_inputs
        }
        # return initial state
        if self.return_initial_state:
            batch_size = images.shape[0]
            X['decoder_state'] = np.zeros([batch_size, self.state_size])

        Y = {
            "output_seqs" : labels
        }

        return X, Y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        if self.shuffle:
            self.dataset.shuffle()


def text2label(text, char2idx):
    return np.array([char2idx[char] for char in text])


class JAMOSeq2SeqGenerator(Sequence):
    "초성 중성 종성 각각 나누어서 Return"

    def __init__(self, dataset, batch_size=32,
                 blank_value=-1, shuffle=True):
        """
        Initialization

        param
        :param dataset : instance of class 'OCRDataset'
        :param batch_size : the number of batch
        :param blank_value : the value of `blank` label
        :param shuffle : whether shuffle dataset or not
        """
        if isinstance(dataset, dict):
            dataset = OCRDataset(**dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.blank_value = blank_value
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return len(self.dataset) // self.batch_size

    def __getitem__(self, index):
        "Generator one batch of dataset"
        images, texts = self.dataset[self.batch_size * index:
                                     self.batch_size * (index + 1)]

        max_len = max([len(text) for text in texts]) + 1
        unicode_arr = np.ones((self.batch_size, max_len), dtype=np.int) * -1
        for idx, text in enumerate(texts):
            unicode_arr[idx, :len(text)] = np.array([ord(char) for char in text])
            unicode_arr[idx, len(text)] = ord('\n')
        decode_inputs = np.roll(unicode_arr, 1, axis=1)
        decode_inputs[:, 0] = ord('\n')
        decode_inputs = decode_inputs.astype(np.int32)

        X = {
            "images": images,
            "output_sequences": unicode_arr
        }
        return (X, )

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        if self.shuffle:
            self.dataset.shuffle()


class DataGenerator(Sequence):
    "Generates Text Recognition Dataset for Keras"

    def __init__(self, dataset, batch_size=32, blank_value=-1, shuffle=True):
        "Initialization"
        self.dataset = dataset
        self.batch_size = batch_size
        self.blank_value = blank_value
        self.shuffle = shuffle
        self.num_classes = self.dataset.labels.max() + 1  # With Blank
        self.max_length = self.dataset.digit_range[-1] - 1
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return len(self.dataset) // self.batch_size

    def __getitem__(self, index):
        "Generator one batch of dataset"
        images, labels, _ = self.dataset[self.batch_size * index:
                                         self.batch_size * (index + 1)]
        # Add Channel axis (batch, width, height) -> (batch, width, height, 1)
        batch_images = images[..., np.newaxis]

        # label sequence
        batch_labels = np.ones([self.batch_size, self.max_length+1], np.int32)
        batch_labels *= self.blank_value  # EOS Token value : -1
        for idx, label in enumerate(labels):
            batch_labels[idx, :len(label)] = label

        return batch_images, batch_labels

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        if self.shuffle:
            self.dataset.shuffle()


class Seq2SeqGenerator(Sequence):
    "Generates Text Recognition Dataset for Keras"

    def __init__(self, dataset, batch_size=32, blank_value=-1, shuffle=True,
                 return_initial_state=True, state_size=512):
        "Initialization"
        self.dataset = dataset
        self.batch_size = batch_size
        self.blank_value = blank_value
        self.shuffle = shuffle
        self.return_initial_state = return_initial_state
        self.state_size = state_size

        self.num_classes = self.dataset.labels.max() + 1  # With Blank
        self.max_length = self.dataset.digit_range[-1] - 1
        self.on_epoch_end()

        self.idx2char = {i: str(i) for i in range(10)}
        self.idx2char[10] = '<EOS>'
        self.idx2char[-1] = ""


    def __len__(self):
        "Denotes the number of batches per epoch"
        return len(self.dataset) // self.batch_size

    def __getitem__(self, index):
        "Generator one batch of dataset"
        images, labels, _ = self.dataset[self.batch_size * index:
                                         self.batch_size * (index + 1)]
        # Add Channel axis (batch, width, height) -> (batch, width, height, 1)
        batch_images = images[..., np.newaxis]

        # label sequence
        batch_labels = np.ones([self.batch_size, self.max_length+1], np.int32)
        batch_labels *= -1  # BLANK Token value : -1
        for idx, label in enumerate(labels):
            batch_labels[idx, :len(label)] = label
            batch_labels[idx, len(label)] = self.num_classes # <EOS> Token

        target_inputs = np.roll(batch_labels, 1, axis=1)
        target_inputs[:, 0] = self.num_classes # <EOS> Token
        target_inputs[target_inputs==-1] = self.num_classes # <EOS> Token

        X = {
            "images" : batch_images,
            "decoder_inputs" : target_inputs
        }
        # return initial state
        if self.return_initial_state:
            batch_size = batch_images.shape[0]
            X['decoder_state'] = np.zeros([batch_size, self.state_size])

        Y = {
            "output_seqs": batch_labels
        }

        return X, Y

    def convert2text(self, arr):
        df = pd.DataFrame(arr)
        df = df.applymap(lambda x: self.idx2char[x])
        texts = df.apply(lambda x: "".join(x), axis=1).values
        return texts

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        if self.shuffle:
            self.dataset.shuffle()