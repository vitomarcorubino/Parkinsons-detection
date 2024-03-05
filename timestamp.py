#!/usr/bin/env python3
import json
import wave
import sys

from vosk import Model, KaldiRecognizer, SetLogLevel
from pydub import AudioSegment

def slice_and_export_audio(audio_path, start_times, end_times):
    # Load audio file using pydub
    audio = AudioSegment.from_wav(audio_path)

    i = 0
    while i < len(start_times) and i < len(end_times):
        # Get start and end times in milliseconds
        start_time_ms = start_times[i] * 1000
        end_time_ms = end_times[i] * 1000

        # Slice the audio
        sliced_audio = audio[start_time_ms:end_time_ms]

        # Export the sliced audio
        sliced_audio.export(f"audio/sliced_audio{i}.wav", format="wav")

        i = i + 1


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

# Load audio file using pydub
audio = AudioSegment.from_wav("audio/italianoMono.wav")

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

            slice_and_export_audio("audio/italianoMono.wav", start_times, end_times)


            i = i + 1

        print(start_times)
        print(end_times)

    data = wf.readframes(4000)
