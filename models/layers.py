"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
from tensorflow.keras.layers import Conv2D, Layer
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Bidirectional, GRU
from tensorflow.keras.layers import Softmax, Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Concatenate
from tensorflow.keras import backend as K
from models.normalization import GroupNormalization
from .jamo import 초성, 중성, 종성
import tensorflow as tf

"""
Reference : 

1. CRNN Paper, 
   An End-to-End Trainable Neural Network for Image-based Sequence Recognition 
   and its Applications to Scene Text Recognition

2. GRU Attention Decoder,
   Robust Scene Text Recognition With Automatic Rectification

3. KaKao OCR Blog,
   카카오 OCR 시스템 구성과 모델, https://brunch.co.kr/@kakao-it/318
"""


class ConvFeatureExtractor(Layer):
    """
    CRNN 중 Convolution Layers에 해당하는 Module Class

    | Layer Name | #maps | Filter | Stride | Padding | activation |
    | ----       | ---   | -----  | ------ | -----   | ---- |
    | conv1      | 64    | (3,3)  | (1,1)  | same    | relu |
    | maxpool1   | -     | (2,2)  | (2,2)  | same    | -    |
    | conv2      | 128   | (3,3)  | (1,1)  | same    | relu |
    | maxpool2   | -     | (2,2)  | (2,2)  | same    | -    |
    | conv3      | 256   | (3,3)  | (1,1)  | same    | relu |
    | conv4      | 256   | (3,3)  | (1,1)  | same    | relu |
    | maxpool3   | -     | (2,1)  | (2,1)  | same    | -    |
    | batchnorm1 | -     | -      | -      | -       | -    |
    | conv5      | 512   | (3,3)  | (1,1)  | same    | relu |
    | batchnorm2 | -     | -      | -      | -       | -    |
    | conv6      | 512   | (3,3)  | (1,1)  | same    | relu |
    | maxpool4   | -     | (2,1)  | (2,1)  | same    | -    |
    | conv7      | 512   | (3,3)  | (1,1)  | valid   | relu |


    특징
     1. Maxpool3와 Maxpool4는 Rectangular Pooling의 형태를 띄고 있음.

    """
    def __init__(self, n_hidden=64, **kwargs):
        self.n_hidden = n_hidden
        super().__init__(**kwargs)
        # for builing Layer and Weight
        self.conv1 = Conv2D(n_hidden, (3, 3), activation='relu', padding='same')
        self.maxpool1 = MaxPooling2D((2, 2), (2, 2), padding='same')
        self.conv2 = Conv2D(n_hidden*2, (3, 3), activation='relu', padding='same')
        self.maxpool2 = MaxPooling2D((2, 2), (2, 2), padding='same')
        self.conv3 = Conv2D(n_hidden*4, (3, 3), activation='relu', padding='same')
        self.conv4 = Conv2D(n_hidden*4, (3, 3), activation='relu', padding='same')
        self.maxpool3 = MaxPooling2D((2, 1), (2, 1), padding='same')
        self.batchnorm1 = BatchNormalization()
        self.conv5 = Conv2D(n_hidden*8, (3, 3), activation='relu', padding='same')
        self.batchnorm2 = BatchNormalization()
        self.conv6 = Conv2D(n_hidden*8, (3, 3), activation='relu', padding='same')
        self.maxpool4 = MaxPooling2D((2, 1), (2, 1), padding='same')
        self.conv7 = Conv2D(n_hidden*8, (2, 2), activation='relu', padding='valid')

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool3(x)
        x = self.batchnorm1(x)
        x = self.conv5(x)
        x = self.batchnorm2(x)
        x = self.conv6(x)
        x = self.maxpool4(x)
        return self.conv7(x)

    def get_config(self):
        config = {
            "n_hidden": self.n_hidden
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ResidualConvFeatureExtractor(Layer):
    """
    Residual Block & KAKAO OCR Like Text Recognition Model
    [ conv2d - layer norm - conv2d - layer norm - maxpool ] * 3

    특징
     1. Batch Normalization 대신 Group Normalization을 이용
     2. KAKAO와 달리 Block 수를 4개로 진행 (보다 넓은 범위를 탐색하기 위함)
     3. Residual Block을 두어서 보다 빠르게 학습가능하도록 설정

    """
    def __init__(self, n_hidden=64, **kwargs):
        self.n_hidden = n_hidden
        super().__init__(**kwargs)
        # for builing Layer and Weight
        self.conv1_1 = Conv2D(n_hidden, (3, 3), activation='relu', padding='same')
        self.norm1_1 = GroupNormalization(groups=n_hidden//4)
        self.conv1_2 = Conv2D(n_hidden, (3, 3), padding='same')
        self.norm1_2 = GroupNormalization(groups=n_hidden//4)
        self.maxpool1 = MaxPooling2D((2, 2), (2, 2), padding='same')

        self.conv2_skip = Conv2D(n_hidden*2, (1, 1), activation='relu', padding='same')
        self.conv2_1 = Conv2D(n_hidden*2, (3, 3), activation='relu', padding='same')
        self.norm2_1 = GroupNormalization(groups=n_hidden//4)
        self.conv2_2 = Conv2D(n_hidden*2, (3, 3), padding='same')
        self.norm2_2 = GroupNormalization(groups=n_hidden//4)
        self.maxpool2 = MaxPooling2D((2, 2), (2, 2), padding='same')

        self.conv3_skip = Conv2D(n_hidden * 4, (1, 1), activation='relu', padding='same')
        self.conv3_1 = Conv2D(n_hidden*4, (3, 3), activation='relu', padding='same')
        self.norm3_1 = GroupNormalization(groups=n_hidden//4)
        self.conv3_2 = Conv2D(n_hidden*4, (3, 3), padding='same')
        self.norm3_2 = GroupNormalization(groups=n_hidden//4)
        self.maxpool3 = MaxPooling2D((2, 1), (2, 1), padding='same')

        self.conv4_skip = Conv2D(n_hidden * 4, (1, 1), activation='relu', padding='same')
        self.conv4_1 = Conv2D(n_hidden*4, (3, 3), activation='relu', padding='same')
        self.norm4_1 = GroupNormalization(groups=n_hidden//4)
        self.conv4_2 = Conv2D(n_hidden*4, (3, 3), padding='same')
        self.norm4_2 = GroupNormalization(groups=n_hidden//4)
        self.maxpool4 = MaxPooling2D((2, 1), (2, 1), padding='same')
        self.built = True

    def call(self, inputs, **kwargs):
        x = self.conv1_1(inputs)
        x = self.norm1_1(x)
        x = self.conv1_2(x)
        x = self.norm1_2(x)
        x = self.maxpool1(x)

        skip = self.conv2_skip(x)
        x = self.conv2_1(x)
        x = self.norm2_1(x)
        x = self.conv2_2(x)
        x = self.norm2_2(x)
        x = K.relu(skip + x)
        x = self.maxpool2(x)

        skip = self.conv3_skip(x)
        x = self.conv3_1(x)
        x = self.norm3_1(x)
        x = self.conv3_2(x)
        x = self.norm3_2(x)
        x = K.relu(skip + x)
        x = self.maxpool3(x)

        skip = self.conv4_skip(x)
        x = self.conv4_1(x)
        x = self.norm4_1(x)
        x = self.conv4_2(x)
        x = self.norm4_2(x)
        x = K.relu(skip + x)
        x = self.maxpool4(x)
        return x

    def get_config(self):
        config = {
            "n_hidden": self.n_hidden
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Map2Sequence(Layer):
    """ CNN Layer의 출력값을 RNN Layer의 입력값으로 변환하는 Module Class
    Transpose & Reshape을 거쳐서 진행

    CNN output shape  ->  RNN Input Shape

    (batch size, height, width, channels)
    -> (batch size, width, height * channels)

    * Caution
        이 때 batch size와 width는 입력 데이터에 따라 변하는 Dynamic Shape이고
        height와 channels는 Convolution Layer에 의해 정해진 Static Shape이다

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        # Get the Dynamic Shape
        shape = tf.shape(inputs)
        batch_size = shape[0]
        f_width = shape[2]

        # Get the Static Shape
        _, f_height, _, f_num = inputs.shape.as_list()
        inputs = K.permute_dimensions(inputs, (0, 2, 1, 3))
        return tf.reshape(inputs,
                          shape=[batch_size, f_width,
                                 f_height * f_num])


class BGRUEncoder(Layer):
    """
    CRNN 중 Recurrent Layers에 해당하는 Module Class
    Convolution Layer의 Image Feature Sequence를 Encoding하여,
    우리가 원하는 Text Feature Sequence로 만듦

    | Layer Name | #Hidden Units |
    | ----       | ------ |
    | Bi-GRU1   | 256    |
    | Bi-GRU2   | 256    |

    """
    def __init__(self, n_units=256, **kwargs):
        self.n_units = n_units
        super().__init__(**kwargs)
        self.gru1 = Bidirectional(GRU(n_units, return_sequences=True))
        self.gru2 = Bidirectional(GRU(n_units, return_sequences=True, return_state=True))

    def call(self, inputs, **kwargs):
        x = self.gru1(inputs)
        state_outputs, forward_s, backward_s = self.gru2(x)
        return state_outputs, forward_s, backward_s

    def get_config(self):
        config = {
            "n_units": self.n_units
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CTCDecoder(Layer):
    """
    CRNN 중 Transcription Layer에 해당하는 Module Class

    * beam_width :
      클수록 탐색하는 Candidate Sequence가 많아져 정확도는 올라가나,
      연산 비용도 함께 올라가기 때문에 ACC <-> Speed의 Trade-Off 관계에 있음

    """
    def __init__(self, beam_width=100, **kwargs):
        self.beam_width = beam_width
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        shape = tf.shape(inputs)
        batch_size = shape[0]
        max_length = shape[1, None]
        input_length = tf.tile(max_length, [batch_size])

        prediction, scores = K.ctc_decode(inputs, input_length, beam_width=self.beam_width)
        return [prediction, scores]

    def get_config(self):
        config = {
            "beam_width": self.beam_width
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DotAttention(Layer):
    """ General Dot-Product Attention Network (Luong, 2015)

    * n_state :
       if n_state is None, Dot-Product Attention(s_t * h_i)
       if n_state is number, general Dot-Product Attention(s_t * W_a * h_i)

    """
    def __init__(self, n_state=None, **kwargs):
        super().__init__(**kwargs)
        self.n_state = n_state
        if isinstance(self.n_state, int):
            self.key_dense = Dense(self.n_state)
            self.val_dense = Dense(self.n_state)

    def call(self, inputs, **kwargs):
        states_encoder = inputs[0]
        states_decoder = inputs[1]

        # (0) adjust the size of encoder state to the size of decoder state
        if isinstance(self.n_state, int):
            # Key Vector와 Value Vector을 다르게 둚
            key_vector = self.key_dense(states_encoder)
            val_vector = self.val_dense(states_encoder)
        else:
            key_vector = states_encoder
            val_vector = states_encoder

        # (1) Calculate Score
        expanded_states_encoder = key_vector[:, None, ...]
        # >>> (batch size, 1, length of encoder sequence, num hidden)
        expanded_states_decoder = states_decoder[..., None, :]
        # >>> (batch size, length of decoder sequence, 1, num hidden)
        score = K.sum(expanded_states_encoder * expanded_states_decoder,
                      axis=-1)
        # >>> (batch size, length of decoder input, length of encoder input)
        # (2) Normalize score
        attention = Softmax(axis=-1, name='attention')(score)

        # (3) Calculate Context Vector
        expanded_val_vector = val_vector[:, None, ...]
        context = K.sum(expanded_val_vector * attention[..., None], axis=2)
        # >>> (batch size, length of decoder input, num hidden)
        return context, attention

    def get_config(self):
        config = {
            "n_state": self.n_state
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class JamoEmbedding(Layer):
    """ 한글 Unicode 번호를 Jamo 별로 Decompose 후 각각 Embedding 하는 Module Class
    """

    def __init__(self, n_embed=16, **kwargs):
        super().__init__(**kwargs)
        self.n_embed = n_embed
        # input_dim : 자모자 갯수 + <EOS> Token
        self.초성_layer = Embedding(input_dim=len(초성) + 1,
                                   output_dim=n_embed)
        self.중성_layer = Embedding(input_dim=len(중성) + 1,
                                   output_dim=n_embed)
        self.종성_layer = Embedding(input_dim=len(종성) + 1,
                                   output_dim=n_embed)

    def call(self, inputs, **kwargs):
        # (1) decompose : 자모자로 분리하기
        inputs = tf.cast(inputs, dtype=tf.int32)
        초성_arr, 중성_arr, 종성_arr = JamoDeCompose()(inputs)

        # (2) embed : Embedding Layer 통과하기
        초성_embed = self.초성_layer(초성_arr)
        중성_embed = self.중성_layer(중성_arr)
        종성_embed = self.종성_layer(종성_arr)

        # (3) concat : 하나의 embedding Vector로 쌓기
        return K.concatenate([초성_embed, 중성_embed, 종성_embed],
                             axis=-1)

    def get_config(self):
        config = {
            "n_embed": self.n_embed
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class JamoCompose(Layer):
    """ 자모자 Compose하여 Unicode 숫자로 바꾸어주는 Module Layer
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        초성_arr, 중성_arr, 종성_arr = tf.split(inputs,
                                            [len(초성)+1,len(중성)+1,len(종성)+1],
                                            axis=-1)
        초성_arr = tf.cast(tf.argmax(초성_arr, axis=-1),dtype=K.floatx())
        중성_arr = tf.cast(tf.argmax(중성_arr, axis=-1),dtype=K.floatx())
        종성_arr = tf.cast(tf.argmax(종성_arr, axis=-1),dtype=K.floatx())

        eos_mask = (tf.less(초성_arr, len(초성)) |
                    tf.less(중성_arr, len(중성)) |
                    tf.less(종성_arr, len(종성)))
        unicode_arr = tf.cast((초성_arr * 21 + 중성_arr) * 28 + 종성_arr + 44032,dtype=tf.int32)
        unicode_arr = tf.where(eos_mask,
                               unicode_arr,
                               tf.ones_like(unicode_arr)*ord('\n'))
        return unicode_arr


class JamoDeCompose(Layer):
    """ 자모자 Unicode를 Decompose하여 초성/중성/종성별로 나누어주는 함수 Module Layer
    """
    def call(self, inputs, **kwargs):
        # # <EOS> Token & <BLANK> Token mask
        mask = (tf.not_equal(inputs, ord('\n')) & tf.not_equal(inputs, -1))

        초성_arr = ((inputs - 44032) // 28) // 21
        초성_arr = tf.where(mask, 초성_arr, tf.ones_like(초성_arr)*len(초성))

        중성_arr = ((inputs - 44032) // 28) % 21
        중성_arr = tf.where(mask, 중성_arr, tf.ones_like(중성_arr)*len(중성))

        종성_arr = (inputs - 44032) % 28
        종성_arr = tf.where(mask, 종성_arr, tf.ones_like(종성_arr)*len(종성))
        return 초성_arr, 중성_arr, 종성_arr


class JamoClassifier(Layer):
    """ 자모자 별로 분류하는 Classifier
    자모별로 Dense Layer *2 & Softmax를 둚
    """
    def __init__(self, n_hidden, **kwargs):
        super().__init__(**kwargs)
        self.n_hidden = n_hidden
        self.초성_fc = Dense(n_hidden, activation='relu', name='chosung_fc')
        self.초성_clf = Dense(len(초성)+1, activation='softmax', name='chosung_output')

        self.중성_fc = Dense(n_hidden, activation='relu', name='joongsung_fc')
        self.중성_clf = Dense(len(중성)+1, activation='softmax', name='joongsung_output')

        self.종성_fc = Dense(n_hidden, activation='relu', name='jongsung_fc')
        self.종성_clf = Dense(len(종성)+1, activation='softmax', name='jongsung_output')

        self.concat = Concatenate(axis=-1)
        self.built = True

    def call(self, inputs, **kwargs):
        초성_output = self.초성_clf(self.초성_fc(inputs))
        중성_output = self.중성_clf(self.중성_fc(inputs))
        종성_output = self.종성_clf(self.종성_fc(inputs))
        return self.concat([초성_output, 중성_output, 종성_output])

    def get_config(self):
        config = {
            "n_hidden": self.n_hidden
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TeacherForcing(Layer):
    def call(self, inputs, **kwargs):
        inputs = tf.roll(inputs, shift=1, axis=1)
        eos = tf.ones_like(inputs[:, :1]) * ord('\n')
        return tf.concat([eos, inputs[:, 1:]], axis=1)


__all__ = ["ConvFeatureExtractor",
           "ResidualConvFeatureExtractor",
           "Map2Sequence",
           "BGRUEncoder",
           "CTCDecoder",
           "DotAttention",
           "JamoDeCompose",
           "JamoCompose",
           "JamoEmbedding",
           "JamoClassifier",
           "TeacherForcing"]
