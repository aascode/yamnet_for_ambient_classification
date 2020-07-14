import pyaudio
import numpy as np
import pandas as pd

import os

import tensorflow as tf

import params
import yamnet as yamnet_model


#   Suppress warnings.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#   Load meeting-contextual class waitings.
df = pd.read_csv('yamnet_remappings.csv')

context_matrix = df.iloc[:, 3:9].to_numpy().T
context_classes = df.columns[3:9]

#   Set up YAMNet.
graph = tf.Graph()
with graph.as_default():
    yamnet = yamnet_model.yamnet_frames_model(params)
    yamnet.load_weights('yamnet.h5')
yamnet_classes = yamnet_model.class_names('yamnet_class_map.csv')

#   Open audio stream.
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000,
                input=True, frames_per_buffer=1024)

with graph.as_default():
    try:
        while True:
            # Sample from laptop microphone at 16 kHz.
            data = np.frombuffer(stream.read(16000, exception_on_overflow=False), dtype=np.int16)

            waveform = data / 32768.0  # Convert to [-1.0, +1.0]
            scores, _ = yamnet.predict(np.reshape(waveform, [1, -1]), steps=1)
            os.system('clear')

            prediction = np.sum(scores, axis=0)
            context_predictions = np.dot(context_matrix, prediction)

            for i, class_ in enumerate(context_classes):
                print('{:25s}{:10.12f}'.format(class_, context_predictions[i]))

            # Report the highest-scoring classes and their scores.
            # top_5 = np.argsort(prediction)[::-1][:5]
            # index = np.argsort(prediction)[::-1][0]

            # for index in top_5:
            # print(('  {:12s}:  {:.3f}'.format(yamnet_classes[index], prediction[index])), end=" "* 100 + "\r")

    except KeyboardInterrupt:
        pass

stream.stop_stream()
stream.close()
p.terminate()