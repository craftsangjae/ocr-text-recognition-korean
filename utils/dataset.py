"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
import numpy as np
import pandas as pd
import os
import requests
from tqdm import tqdm
import cairocffi as cairo
from imgaug import augmenters as iaa
import matplotlib.font_manager as fm
from tensorflow.keras.utils import get_file
"""
# All about MNIST Style DataSet

> We include the following dataset list
    - mnist : handwritten digits dataset

    - fashionmnist : dataset of Zalando's article images

    - handwritten : handwritten a ~ z Alphabet dataset

"""
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATASET_DIR = os.path.join(ROOT_DIR,"datasets/")
DOWNLOAD_URL_FORMAT = "https://s3.ap-northeast-2.amazonaws.com/pai-datasets/all-about-mnist/{}/{}.csv"

FONT_LIST = [f.name for f in fm.fontManager.ttflist]
FONT_LIST = list(set([f for f in FONT_LIST if "Nanum" in f and not 'Square' in f]))


class OCRRandomDataset:
    """
    generate OCR dataset for Text Recognition not using Dicitonary word



    :param word_range : 철자의 길이 범위
    :param bg_noise : 가우시안 노이즈의 강도 (0.0~0.5)
    """

    def __init__(self, word_range, font_size,
                 max_data=10000,
                 bg_noise=0.0,
                 affine_noise=(0.0, 0.02),
                 color_noise=(0.0, 0.6),
                 normalize=False,
                 random_shift=True,
                 gray_scale=True):
        self.word_range = word_range
        self.max_word = word_range[1]
        self.max_data = max_data
        self.font_size = font_size
        self.bg_noise = np.clip(bg_noise, 0., 0.5)
        self.gray_scale = gray_scale
        self.aug = iaa.PiecewiseAffine(scale=affine_noise)
        self.normalize = normalize
        self.random_shift = random_shift
        self.color_noise = color_noise
        self.paint_text = lambda word: paint_text(word,
                                                  self.max_word,
                                                  self.font_size,
                                                  self.color_noise,
                                                  self.random_shift)

    def __len__(self):
        # Default Value
        return self.max_data

    def __getitem__(self, index):
        """
        (1) index -> integer일 때, get single item
        (2) index -> np.array | slice, get batch items
        """
        #
        if isinstance(index, int):
            w_range = np.random.randint(self.word_range[0],
                                        self.word_range[1]+1)
            word = "".join([chr(idx)
                            for idx
                            in np.random.randint(ord('가'), ord('힣') + 1, w_range)])
            image = self.paint_text(word)
            if self.bg_noise > 0:
                image = gaussian_noise(image, self.bg_noise)
            image = self.aug.augment_image(image)
            if self.gray_scale:
                image = (0.3 * image[:, :, 0]
                         + 0.59 * image[:, :, 1]
                         + 0.11 * image[:, :, 2])
                image = image[:, :, None]
            image = (image * 255).astype(np.uint8)
            return image, word
        else:
            indices = np.arange(self.max_data)[index]
            images = []
            words = []
            for _ in indices:
                w_range = np.random.randint(self.word_range[0],
                                            self.word_range[1] + 1)
                word_indices = np.random.randint(ord('가'), ord('힣') + 1, w_range)
                word = "".join([chr(idx) for idx in word_indices])
                image = self.paint_text(word)

                if self.bg_noise > 0:
                    noise = np.random.uniform(0, self.bg_noise)
                    image = gaussian_noise(image, noise)

                image = self.aug.augment_image(image)
                if self.gray_scale:
                    image = (0.3 * image[:, :, 0]
                             + 0.59 * image[:, :, 1]
                             + 0.11 * image[:, :, 2])
                    image = image[:, :, None]
                images.append(image)
                words.append(word)

            images = np.stack(images)
            if self.normalize:
                images = (images * 255).astype(np.uint8)
            words = np.array(words)
            return images, words

    def shuffle(self):
        pass


class OCRDataset:
    """
    generate OCR dataset for Text Recognition

    텍스트 Recognition에 필요한 단어집합을 만들어주는 것

    :param words : OCR 단어로 생성할 단어 집합들
    :param bg_noise : 가우시안 노이즈의 강도 (0.0~0.5)
    """

    def __init__(self, words, font_size,
                 bg_noise=0.0,
                 affine_noise=(0.0, 0.02),
                 color_noise=(0.0, 0.6),
                 normalize=False,
                 random_shift=True,
                 gray_scale=True,
                 font_list=FONT_LIST):
        self.words = np.array(words)
        self.max_word = max([len(word) for word in self.words])
        self.font_size = font_size
        self.bg_noise = np.clip(bg_noise, 0., 0.5)
        self.gray_scale = gray_scale
        self.aug = iaa.PiecewiseAffine(scale=affine_noise)
        self.normalize = normalize
        self.random_shift = random_shift
        self.color_noise = color_noise
        self.font_list = font_list
        self.paint_text = lambda word: paint_text(word,
                                                  self.max_word,
                                                  self.font_size,
                                                  self.color_noise,
                                                  self.random_shift,
                                                  self.font_list)
        if isinstance(words,np.ndarray):
            words = words.tolist()
        self.config = {
            "words": words,
            "font_size": font_size,
            "bg_noise": bg_noise,
            "affine_noise": affine_noise,
            "color_noise": color_noise,
            "normalize": normalize,
            "random_shift": random_shift,
            "gray_scale": gray_scale,
            "font_list": font_list
        }

    def __len__(self):
        # 전체 데이터 셋
        return len(self.words)

    def __getitem__(self, index):
        """
        (1) index -> integer일 때, get single item
        (2) index -> np.array | slice, get batch items
        """
        #
        if isinstance(index, int):
            word = self.words[index]
            image = self.paint_text(word)
            if self.bg_noise > 0:
                image = gaussian_noise(image, self.bg_noise)
            image = self.aug.augment_image(image)
            if self.gray_scale:
                image = (0.3 * image[:, :, 0]
                         + 0.59 * image[:, :, 1]
                         + 0.11 * image[:, :, 2])
                image = image[:, :, None]
            if self.normalize:
                image = (image * 255).astype(np.uint8)
            return image, word
        else:
            words = self.words[index]
            images = []
            for word in words:
                image = self.paint_text(word)
                if self.bg_noise > 0:
                    noise = np.random.uniform(0, self.bg_noise)
                    image = gaussian_noise(image, noise)
                image = self.aug.augment_image(image)
                if self.gray_scale:
                    image = (0.3 * image[:, :, 0]
                             + 0.59 * image[:, :, 1]
                             + 0.11 * image[:, :, 2])
                    image = image[:, :, None]
                images.append(image)
            images = np.stack(images)
            if self.normalize:
                images = (images * 255).astype(np.uint8)
            return images, words

    def shuffle(self):
        np.random.shuffle(self.words)


def paint_text(text, max_word=None, font_size=28, color_noise=(0.,0.6), random_shift=True, font_list=FONT_LIST):
    '''
    Text가 그려진 이미지를 만드는 함수
    '''
    if max_word is None:
        # None이면, text의 word 갯수에 맞춰서 생성
        max_word = len(text)
    h = font_size + 12  # 이미지 높이, font_size + padding
    w = font_size * (1 + max_word) + 12  # 이미지 폭

    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, w, h)
    with cairo.Context(surface) as context:
        context.set_source_rgb(1, 1, 1)  # White
        context.paint()

        # Font Style : Random Pick
        context.select_font_face(
                np.random.choice(font_list),
                cairo.FONT_SLANT_NORMAL,
                np.random.choice([cairo.FONT_WEIGHT_BOLD,
                                  cairo.FONT_WEIGHT_NORMAL]))

        context.set_font_size(font_size)
        box = context.text_extents(text)
        border_w_h = (4, 4)
        if box[2] > (w - 2 * border_w_h[1]) or box[3] > (h - 2 * border_w_h[0]):
            raise IOError(('Could not fit string into image.'
                           'Max char count is too large for given image width.'))

        # Random Shift을 통해, 이미지 Augmentation
        max_shift_x = w - box[2] - border_w_h[0]
        max_shift_y = h - box[3] - border_w_h[1]

        if random_shift:
            top_left_x = np.random.randint(0, int(max_shift_x))
            top_left_y = np.random.randint(0, int(max_shift_y))
        else:
            top_left_x = np.random.randint(3, 10) # Default padding
            top_left_y = int(max_shift_y)//2 # Default position = center in heights

        context.move_to(top_left_x - int(box[0]),
                        top_left_y - int(box[1]))

        # Draw Text
        rgb = np.random.uniform(*color_noise, size=3)

        context.set_source_rgb(*rgb)
        context.show_text(text)

    # cairo data format to numpy data format
    buf = surface.get_data()
    text_image = np.frombuffer(buf, np.uint8)
    text_image = text_image.reshape(h, w, 4)
    text_image = text_image[:, :, :3]
    text_image = text_image.astype(np.float32) / 255

    return text_image


def gaussian_noise(image, noise=0.1):
    '''
    이미지에 가우시안 잡음을 넣어주는 함수
    Data Augmentation을 적용하기 위함
    '''
    image = image + np.random.normal(0, noise, size=image.shape)
    return np.clip(image, 0, 1)


class SerializationDataset:
    """
    generate data for Serialization

    이 class는 단순히 숫자를 나열하는 것

    :param dataset : Select one, (mnist, fashionmnist, handwritten)
    :param data_type : Select one, (train, test, validation)
    :param digit : the length of number (몇개의 숫자를 serialize할 것인지 결정)
    :param bg_noise : the background noise of image, bg_noise = (gaussian mean, gaussian stddev)
    :param pad_range : the padding length between two number (두 숫자 간 거리, 값의 범위로 주어 랜덤하게 결정)
    """

    def __init__(self, dataset="mnist", data_type="train",
                 digit=5, bg_noise=(0, 0.2), pad_range=(3, 10)):
        """
        generate data for Serialization

        :param dataset: Select one, (mnist, fashionmnist, handwritten)
        :param data_type: Select one, (train, test, validation)
        :param digit : the length of number
          if digit is integer, the length of number is always same value.
          if digit is tuple(low_value, high_value), the length of number will be determined within the range
        :param bg_noise : the background noise of image, bg_noise = (gaussian mean, gaussian stddev)
        :param pad_range : the padding length between two number (두 숫자 간 거리, 값의 범위로 주어 랜덤하게 결정)
        """
        self.images, self.labels = load_dataset(dataset, data_type)
        if isinstance(digit, int):
            self.digit_range = (digit, digit + 1)
        else:
            self.digit_range = digit
        self.num_data = len(self.labels) // (self.digit_range[1] - 1)
        self.index_list = np.arange(len(self.labels))

        self.bg_noise = bg_noise
        self.pad_range = pad_range

        self.max_length = int((15 + pad_range[1]) * self.digit_range[1])

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        if isinstance(index, int):
            num_digit = np.random.randint(*self.digit_range)
            start_index = (self.digit_range[1] - 1) * index
            digits = self.index_list[start_index:start_index + num_digit]

            digit_images = self.images[digits]
            digit_labels = self.labels[digits].values
            series_image, series_len = self._serialize_random(digit_images)

            return series_image, digit_labels, series_len

        else:
            batch_images, batch_labels, batch_length = [], [], []
            indexes = np.arange(self.num_data)[index]
            for _index in indexes:
                num_digit = np.random.randint(*self.digit_range)
                start_index = (self.digit_range[1] - 1) * _index
                digits = self.index_list[start_index:start_index + num_digit]

                digit_images = self.images[digits]
                digit_labels = self.labels[digits].values
                series_image, series_len = self._serialize_random(digit_images)
                batch_images.append(series_image)
                batch_labels.append(digit_labels)
                batch_length.append(series_len)

            return np.stack(batch_images), \
                batch_labels, \
                np.stack(batch_length)

    def shuffle(self):
        indexes = np.arange(len(self.images))
        np.random.shuffle(indexes)

        self.images = self.images[indexes]
        self.labels = self.labels[indexes]
        self.labels.index = np.arange(len(self.labels))

    def _serialize_random(self, images):
        """
        복수의 이미지를 직렬로 붙임

        :param images:
        :return:
        """
        pad_height = images.shape[1]
        pad_width = np.random.randint(*self.pad_range)

        serialized_image = np.zeros([pad_height, pad_width])
        for image in images:
            serialized_image = self._place_random(image, serialized_image)

        full_image = np.random.normal(*self.bg_noise,
                                      size=(pad_height, self.max_length))

        if serialized_image.shape[1] < self.max_length:
            series_length = serialized_image.shape[1]
            full_image[:, :serialized_image.shape[1]] += serialized_image
        else:
            series_length = full_image.shape[1]
            full_image += serialized_image[:, :full_image.shape[1]]

        full_image = np.clip(full_image, 0., 1.)
        return full_image, series_length

    def _place_random(self, image, serialized_image):
        """
        가운데 정렬된 이미지를 떼어서 재정렬함

        :param image:
        :param serialized_image:
        :return:
        """
        x_min, x_max, _, _ = crop_fit_position(image)
        cropped = image[:, x_min:x_max]

        pad_height = cropped.shape[0]
        pad_width = np.random.randint(*self.pad_range)
        pad = np.zeros([pad_height, pad_width])

        serialized_image = np.concatenate(
            [serialized_image, cropped, pad], axis=1)
        return serialized_image


def crop_fit_position(image):
    """
    get the coordinates to fit object in image

    :param image:
    :return:
    """
    positions = np.argwhere(
        image >= 0.1)  # set the threshold to 0.1 for reducing the noise

    y_min, x_min = positions.min(axis=0)
    y_max, x_max = positions.max(axis=0)

    return np.array([x_min, x_max, y_min, y_max])


def load_dataset(dataset, data_type):
    """
    Load the MNIST-Style dataset
    if you don't have dataset, download the file automatically

    :param dataset: Select one, (mnist, fashionmnist, handwritten)
    :param data_type: Select one, (train, test, validation)
    :return:
    """
    if dataset not in ["mnist", "fashionmnist", "handwritten"]:
        raise ValueError(
            "allowed dataset: mnist, fashionmnist, handwritten")
    if data_type not in ["train", "test", "validation"]:
        raise ValueError(
            "allowed data_type: train, test, validation")

    url = DOWNLOAD_URL_FORMAT.format(dataset, data_type)
    fpath = get_file(f"{dataset}-{data_type}.csv", url)

    df = pd.read_csv(fpath)

    images = df.values[:, 1:].reshape(-1, 28, 28)
    images = images / 255  # normalization, 0~1
    labels = df.label  # label information
    return images, labels


# Download korean word file path
KOR_WORD_FILE_PATH = os.path.join(DATASET_DIR,"wordslist.txt")
if not os.path.exists(KOR_WORD_FILE_PATH):
    response = requests.get('https://github.com/acidsound/korean_wordlist/raw/master/wordslist.txt',
                            stream=True)
    with open(KOR_WORD_FILE_PATH, 'wb') as f:
        for chunk in tqdm(response.iter_content(chunk_size=1024)):
            if chunk:
                f.write(chunk)

if __name__ == '__main__':
    pass
