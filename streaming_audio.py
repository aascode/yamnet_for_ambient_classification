""" streaming_audio.py

    Using YAMNet and user created context-remappings, generates ratings the about environmental audio.

    Uses default sound input (e.g., laptop mic), outputs ratings every 1s.

    Summer 2020.

    Max Henry.
    Music Technology Area, Department of Music Research,
    McGill University.

    yamnet: https://github.com/tensorflow/models/tree/master/research/audioset/yamnet

"""

import pyaudio
import numpy as np
import pandas as pd

import sys
import os


import tensorflow as tf

import params
import yamnet as yamnet_model


#   Check if OS X, then can have nicely formatted output (cleared screen):
if sys.platform == 'darwin':
    is_mac = True

#   Suppress warnings.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#   Load meeting-contextual class weightings.
df = pd.read_csv('yamnet_remappings.csv')
context_matrix = df.iloc[:, 3:9].to_numpy().T
context_classes = df.columns[3:9]

#   Open audio stream.
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000,
                input=True, frames_per_buffer=1024)

#   Set up YAMNet.

graph = tf.Graph()
with graph.as_default():
    yamnet = yamnet_model.yamnet_frames_model(params)
    yamnet.load_weights('yamnet.h5')
    yamnet_classes = yamnet_model.class_names('yamnet_class_map.csv')

    #   Sampling/output loop. (Keyboard interrupt to stop).
    try:
        while True:
            #   Sample 1 sec from laptop microphone at 16 kHz.
            data = np.frombuffer(stream.read(16000, exception_on_overflow=False), dtype=np.int16)

            #   Normalize.
            waveform = data / 32768.0

            scores, _ = yamnet.predict(np.reshape(waveform, [1, -1]), steps=1)

            if is_mac:
                os.system("clear")

            #   Sum sub-frames of predictions, and translate to context-specific ratings.
            prediction = np.sum(scores, axis=0)
            context_predictions = np.dot(context_matrix, prediction)

            #   Print context-specific ratings.
            for i, class_ in enumerate(context_classes):
                print('{:25s}{:10.12f}'.format(class_, context_predictions[i]))
    except KeyboardInterrupt:
        pass

stream.stop_stream()
stream.close()
p.terminate()
