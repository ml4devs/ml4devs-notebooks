# (c) Copyright 2020-2022 Satish Chandra Gupta
# 
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#


# For more explanation, check following blog posts:
#   - https://www.ml4devs.com/articles/how-to-build-python-transcriber-using-mozilla-deepspeech/
#   - https://www.ml4devs.com/articles/speech-recognition-with-python/

import stt
import numpy as np
import os
import pyaudio
import time

# DeepSpeech parameters
DEEPSPEECH_MODEL_DIR = 'coqui-stt-1.0.0-models'
MODEL_FILE_PATH = os.path.join(DEEPSPEECH_MODEL_DIR, 'model.tflite')
SCORER_FILE_PATH = os.path.join(DEEPSPEECH_MODEL_DIR, 'large_vocabulary.scorer')
LM_ALPHA = 0.75
LM_BETA = 1.85
BEAM_WIDTH = 500

# Make DeepSpeech Model
model = stt.Model(MODEL_FILE_PATH)
model.enableExternalScorer(SCORER_FILE_PATH)
model.setScorerAlphaBeta(LM_ALPHA, LM_BETA)
model.setBeamWidth(BEAM_WIDTH)

# Create a Streaming session
stt_stream = model.createStream()

# Encapsulate DeepSpeech audio feeding into a callback for PyAudio
text_so_far = ''
def process_audio(in_data, frame_count, time_info, status):
    global text_so_far
    data16 = np.frombuffer(in_data, dtype=np.int16)
    stt_stream.feedAudioContent(data16)
    text = stt_stream.intermediateDecode()
    if text != text_so_far:
        print('Interim text = {}'.format(text))
        text_so_far = text
    return (in_data, pyaudio.paContinue)

# PyAudio parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_SIZE = 1024

# Feed audio to deepspeech in a callback to PyAudio
audio = pyaudio.PyAudio()
stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK_SIZE,
    stream_callback=process_audio
)

print('Please start speaking, when done press Ctrl-C ...')
stream.start_stream()

try: 
    while stream.is_active():
        time.sleep(0.1)
except KeyboardInterrupt:
    # PyAudio
    stream.stop_stream()
    stream.close()
    audio.terminate()
    print('Finished recording.')
    # DeepSpeech
    text = stt_stream.finishStream()
    print('Final text = {}'.format(text))
