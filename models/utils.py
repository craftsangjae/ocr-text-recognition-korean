"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from .jamo import compose_unicode


def generate_text(images, encoder, decoder, max_length=20):
    # Encoder 결과 계산
    states_encoder_ = encoder.predict(images)

    # Decoder 결과 계산
    batch_size = images.shape[0]
    n_state = decoder.inputs[-1].shape.as_list()[-1]
    prev_inputs = np.ones((batch_size,1)) * ord('\n')
    prev_states = np.zeros((batch_size, n_state))
    result = np.zeros((batch_size, 0))
    counter = 0
    while True:
        states_decoder_, predictions_ = decoder.predict({
            "states_encoder_input" : states_encoder_,
            "decoder_inputs": prev_inputs,
            "decoder_state": prev_states
        })
        prev_states = states_decoder_[:,-1,:]
        prev_inputs = predictions_
        if np.all(prev_inputs == ord('\n')) or counter > max_length:
            break
        counter += 1
        result = np.concatenate([result,predictions_],axis=-1)
    texts = compose_unicode(result.astype(np.int))
    texts = [text[:text.find('\n')] for text in texts]
    return texts


def visualize_result(image, feature_seq, text, states, contexts, attentions, figsize=(12, 15), ):
    fig = plt.figure(figsize=(12, 15), dpi=200)
    visualized = {}
    ### Draw Image
    image = (image * 255).astype(np.uint8)
    ax = plt.subplot2grid((18, 8), (0, 0), rowspan=2, colspan=4, fig=fig)
    ax.set_title(f"예측 : {text}")
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    visualized['image'] = image

    ### Draw Feature Sequences
    feature_seq = ((feature_seq - feature_seq.min(axis=1, keepdims=True))
                   / (feature_seq.max(axis=1, keepdims=True)
                      - feature_seq.min(axis=1, keepdims=True) + 1e-10))
    h, w = image.shape
    feature_seq = (feature_seq * 255).astype(np.uint8)
    feature_seq = cv2.resize(feature_seq.transpose(), (w, h))
    ax = plt.subplot2grid((18, 8), (0, 4), rowspan=2, colspan=4, fig=fig)
    ax.set_title('Feature Sequences')
    ax.imshow(feature_seq, cmap='jet')
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    visualized['feature_seq'] = feature_seq

    ### Draw State
    ax = plt.subplot2grid((18, 8), (2, 0), fig=fig)
    ax.text(0.5, 0.5, "state")
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    visualized['states'] = []
    for i, state in enumerate(states[:-1], 1):
        state = state.reshape(16, 16)
        state = (state - state.min()) / (state.max() - state.min())
        state = (state * 255).astype(np.uint8)
        visualized['states'].append(state)
        ax = plt.subplot2grid((18, 8), (2, i), fig=fig)
        ax.imshow(state, cmap='jet')
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])

    ### Draw Context
    ax = plt.subplot2grid((18, 8), (3, 0), fig=fig)
    ax.text(0.5, 0.5, "context")
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    visualized['contexts'] = []
    for i, context in enumerate(contexts, 1):
        context = context.reshape(16, 16)
        context = ((context - context.min()) /
                   (context.max() - context.min()))
        context = (context * 255).astype(np.uint8)
        visualized['contexts'].append(context)

        ax = plt.subplot2grid((18, 8), (3, i), fig=fig)
        ax.imshow(context, cmap='jet')
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])

    ### Draw Foreground
    visualized['attentions'] = []
    visualized['img_concat'] = []
    image_fg = cv2.cvtColor((image * 255).astype(np.uint8),
                            cv2.COLOR_GRAY2RGB)
    for i, attend in enumerate(attentions):
        # Attention Image
        attend_bg = cv2.resize((attend * 255).astype(np.uint8), (w, h))
        ax = plt.subplot2grid((18, 8), (2 * i + 4, 0),
                              rowspan=2, colspan=4, fig=fig)
        ax.imshow(attend_bg, cmap='jet')
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        visualized['attentions'].append(attend_bg)

        # Image Concatenation
        attend_bg = cv2.blur(attend_bg, (20, 1))
        img_concat = np.concatenate([image_fg,
                                     attend_bg[..., None]],
                                    axis=-1)
        ax = plt.subplot2grid((18, 8), (2 * i + 4, 4),
                              rowspan=2, colspan=4, fig=fig)
        ax.imshow(img_concat, cmap='gray')
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])

        visualized['img_concat'].append(img_concat)

    plt.show()
    return visualized