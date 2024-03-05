#!/usr/bin/env python3
import json
import wave
import sys

from vosk import Model, KaldiRecognizer, SetLogLevel
from pydub import AudioSegment


def slice_and_export_audio(audio_path, start_times, end_times, number_of_words):
    # Load audio file using pydub
    audio = AudioSegment.from_wav(audio_path)

    i = 0
    while i < len(start_times):
        # If there are less than number_of_words left, slice until the last word
        if i + number_of_words >= len(end_times):
            # Get the last end time if there are less than number_of_words left
            end_time_ms = end_times[-1] * 1000
        else:
            end_time_ms = end_times[i + number_of_words] * 1000

        # Get start time in milliseconds
        start_time_ms = start_times[i] * 1000

        # Slice the audio
        sliced_audio = audio[start_time_ms:end_time_ms]

        # Export the sliced audio
        sliced_audio.export(f"audio/sliced_audio{i // number_of_words}.wav", format="wav")

        i = i + number_of_words


# Source: https://github.com/alphacep/vosk-api/blob/master/python/example/test_simple.py
# You can set log level to -1 to disable debug messages
SetLogLevel(0)

wf = wave.open("audio/italianoMono.wav", "rb")
if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
    print("Audio file must be WAV format mono PCM.")
    sys.exit(1)

model = Model(lang="it")

# You can also init model by name or with a folder path
# model = Model(model_name="vosk-model-en-us-0.21")
# model = Model("models/en")

rec = KaldiRecognizer(model, wf.getframerate())
rec.SetWords(True)
rec.SetPartialWords(True)

start_times = []
end_times = []

data = wf.readframes(4000)

while len(data) > 0:
    if rec.AcceptWaveform(data):
        result = rec.Result()

        print(result)

        # Convert the result to a JSON object
        result_json = json.loads(result)

        i = 0
        while i < len(result_json['result']):
            # Get the start and end times of the spoken word
            word_info = result_json['result'][i]

            # Get the word start and end times
            start_time = word_info['start']
            end_time = word_info['end']

            start_times.append(start_time)
            end_times.append(end_time)

            slice_and_export_audio("audio/italianoMono.wav", start_times, end_times, 6)

            i = i + 1

        print(start_times)
        print(end_times)

    data = wf.readframes(4000)
