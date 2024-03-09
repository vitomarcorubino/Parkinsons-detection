import os
import json
import wave
import audioFileFormat # Module to handle audio file format and conversion to wav mono pcm
import languageDetection # Module to detect the language of the audio file
import trimming # Module to trim the audio file

from vosk import Model, KaldiRecognizer, SetLogLevel

# Source: https://github.com/alphacep/vosk-api/blob/master/python/example/test_simple.py
# -1 to disable logging messages, 0 to enable them
SetLogLevel(-1)

file_path = "audio/italiano.wav"

wf = wave.open(file_path, "rb")
if not audioFileFormat.is_mono_pcm(wf):
    print("Audio file must be WAV format mono PCM.")
    print("Converting to mono PCM...")
    audioFileFormat.convert_to_mono_pcm(file_path)
    print("Conversion completed.")

    # Get the original file name without extension
    original_file_name = os.path.splitext(file_path)[0]

    # Create the output file name
    file_path = f"{original_file_name}_mono_pcm.wav"

    # Open the converted file
    wf = wave.open(file_path, "rb")

# Get the language id of the audio file, such as "it" for italian
lang_id = languageDetection.get_language_id(file_path)

# Load the language model for the detected language
model = Model(lang=lang_id)

# Create a recognizer object using the language model and the sample rate of the audio file
rec = KaldiRecognizer(model, wf.getframerate())
rec.SetWords(True) # Enable the result to include the words
rec.SetPartialWords(True) # Enable the result to include the partial words

text = "" # The text of the audio
words = [] # The tokenized words of the audio
start_times = [] # The start times of the spoken words
end_times = [] # The end times of the spoken words

data = wf.readframes(4000) # Read the audio file in chunks of 4000 frames

while len(data) > 0:
    if rec.AcceptWaveform(data):
        result = rec.Result()

        # Convert the result to a JSON object
        result_json = json.loads(result)

        text = result_json['text']
        i = 0
        while i < len(result_json['result']):
            # Get the start and end times of the spoken word
            word_info = result_json['result'][i]

            # Get the word start and end times
            word = word_info['word']
            start_time = word_info['start']
            end_time = word_info['end']

            words.append(word)
            start_times.append(start_time)
            end_times.append(end_time)

            i = i + 1

        trimming.trim_on_descending_waveform(file_path, start_times, end_times, words, 4)

        print(text)
        print(words)
        print(start_times)
        print(end_times)

    data = wf.readframes(4000)
