"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
import numpy as np

# 한글 자모자를 인덱스로 만드는 Map 구현
초성 = (
    u'ㄱ', u'ㄲ', u'ㄴ', u'ㄷ', u'ㄸ', u'ㄹ', u'ㅁ', u'ㅂ', u'ㅃ', u'ㅅ',
    u'ㅆ', u'ㅇ', u'ㅈ', u'ㅉ', u'ㅊ', u'ㅋ', u'ㅌ', u'ㅍ', u'ㅎ'
)

중성 = (
    u'ㅏ', u'ㅐ', u'ㅑ', u'ㅒ', u'ㅓ', u'ㅔ', u'ㅕ', u'ㅖ', u'ㅗ', u'ㅘ',
    u'ㅙ', u'ㅚ', u'ㅛ', u'ㅜ', u'ㅝ', u'ㅞ', u'ㅟ', u'ㅠ', u'ㅡ', u'ㅢ', u'ㅣ'
)

종성 = (
    u'', u'ㄱ', u'ㄲ', u'ㄳ', u'ㄴ', u'ㄵ', u'ㄶ', u'ㄷ', u'ㄹ', u'ㄺ',
    u'ㄻ', u'ㄼ', u'ㄽ', u'ㄾ', u'ㄿ', u'ㅀ', u'ㅁ', u'ㅂ', u'ㅄ', u'ㅅ',
    u'ㅆ', u'ㅇ', u'ㅈ', u'ㅊ', u'ㅋ', u'ㅌ', u'ㅍ', u'ㅎ'
)


def decompose_jamo(text):
    onsets = []
    nucleuses = []
    codas = []
    for char in text:
        onset_idx = ((ord(char) - 44032) // 28) // 21
        nucleus_idx = ((ord(char) - 44032) // 28) % 21
        coda_idx = (ord(char) - 44032) % 28
        onsets.append(onset_idx)
        nucleuses.append(nucleus_idx)
        codas.append(coda_idx)

    return np.array(onsets), np.array(nucleuses), np.array(codas)


def compose_jamo(초성_arr, 중성_arr, 종성_arr):
    text = ""
    for onset_idx, nucleus_idx, coda_idx in zip(초성_arr, 중성_arr, 종성_arr):
        if (onset_idx >= len(초성)
                or nucleus_idx >= len(중성)
                or coda_idx >= len(종성)):
            break
        if (onset_idx < 0
                or nucleus_idx < 0
                or coda_idx < 0):
            continue
        text += chr(int((onset_idx * 21 + nucleus_idx) * 28 + coda_idx + 44032))
    return text


def compose_unicode(unicode_arr):
    if unicode_arr.ndim == 1:
        unicode_arr = unicode_arr[None]
    texts = []
    for unicode_vec in unicode_arr:
        text = ""
        for vec in unicode_vec:
            if vec != -1:
                text+=chr(vec)
        texts.append(text)
    return texts
