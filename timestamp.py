#!/usr/bin/env python3
import os
import json
import wave
import languageDetection
import numpy as np
from scipy.io.wavfile import read, write

from vosk import Model, KaldiRecognizer, SetLogLevel
from pydub import AudioSegment


def is_mono_pcm(wav_file):
    if wav_file.getnchannels() == 1 and wav_file.getsampwidth() == 2 and wav_file.getcomptype() == "NONE":
        return True
    else:
        return False


def convert_to_mono_pcm(wav_file_path):
    audio = AudioSegment.from_wav(wav_file_path)

    # Convert to mono
    mono_audio = audio.set_channels(1)

    # Set sample width to 2
    mono_audio = mono_audio.set_sample_width(2)

    # Get the original file name without extension
    original_file_name = os.path.splitext(wav_file_path)[0]

    # Create the output file name
    output_file_name = f"{original_file_name}_mono_pcm.wav"

    # Export as PCM WAV
    mono_audio.export(output_file_name, format="wav")

def trim_on_descending_waveform(audio_path, start_times, end_times, number_of_words, buffer_time=0.00, threshold=0.1, end_buffer=0.075):
    # Load audio file
    sample_rate, audio_data = read(audio_path)

    # Convert audio data to numpy array
    audio_data = np.array(audio_data)

    # Initialize counter
    i = 0

    # Iterate over start and end times
    while i < len(start_times):
        # If there are less than number_of_words left, trim until the last word
        if i + number_of_words >= len(end_times):
            # Get the last end time if there are less than number_of_words left
            end_sample = int((end_times[-1] - buffer_time) * sample_rate + end_buffer * sample_rate)
        else:
            end_sample = int((end_times[i + number_of_words - 1] - buffer_time) * sample_rate + end_buffer * sample_rate)

        # Get start sample
        start_sample = int(start_times[i] * sample_rate)

        # Ensure end_sample doesn't exceed audio length
        end_sample = min(end_sample, len(audio_data))

        # Get the audio segment
        segment = audio_data[start_sample:end_sample]

        # Find the points where the amplitude decreases
        decrease_points = np.where(np.diff(segment) < 0)[0]

        # If there are decrease points, trim the segment at the first decrease point
        if len(decrease_points) > 0 and decrease_points[0] > threshold * len(segment):
            segment = segment[:decrease_points[0]]

        # Export the trimmed segment
        write(f"audio/trimmed_audio{i // number_of_words}.wav", sample_rate, segment)

        # Increment counter
        i = i + number_of_words

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

file_path = "audio/italiano.wav"

wf = wave.open(file_path, "rb")
if not is_mono_pcm(wf):
    print("Audio file must be WAV format mono PCM.")
    print("Converting to mono PCM...")
    convert_to_mono_pcm(file_path)
    print("Conversion completed.")

    # Get the original file name without extension
    original_file_name = os.path.splitext(file_path)[0]

    # Create the output file name
    file_path = f"{original_file_name}_mono_pcm.wav"

    # Open the converted file
    wf = wave.open(file_path, "rb")

    # sys.exit(1)


lang_id = languageDetection.get_language_id(file_path)

model = Model(lang=lang_id)

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

            # slice_and_export_audio(file_path, start_times, end_times, 1)

            trim_on_descending_waveform(file_path, start_times, end_times, 4)

            i = i + 1

        print(start_times)
        print(end_times)

    data = wf.readframes(4000)
