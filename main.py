# Source: https://github.com/alphacep/vosk-api/blob/master/python/example/test_simple.py
import os
import json
import wave
import audioFileFormat  # Module to handle audio file format and conversion to wav mono pcm
import languageDetection  # Module to detect the language of the audio file
import trimming  # Module to trim the audio file

from vosk import Model, KaldiRecognizer, SetLogLevel

file_path = "audio/italiano.wav"  # The path to the audio file
number_of_words = 4  # The number of words to consider for each trimming operation

# -1 to disable logging messages, 0 to enable them
SetLogLevel(-1)

wf = wave.open(file_path, "rb")
if not audioFileFormat.is_mono_pcm(wf):
    print("\nAudio file must be WAV format mono PCM.")
    print("Converting to mono PCM...")
    output_folder_converted = "mono_pcm"
    audioFileFormat.convert_to_mono_pcm(file_path, output_folder_converted)

    # Get the original file name without extension
    original_file_name = os.path.splitext(os.path.basename(file_path))[0]

    # Create the output file name
    file_path = f"{output_folder_converted}/{original_file_name}_mono_pcm.wav"

    print("Conversion completed. File stored into: " + file_path)

    # Open the converted file
    wf = wave.open(file_path, "rb")

# Get the language id of the audio file, such as "it" for italian
lang_id = languageDetection.get_language_id(file_path)

# Load the language model for the detected language
model = Model(lang=lang_id)

# Create a recognizer object using the language model and the sample rate of the audio file
rec = KaldiRecognizer(model, wf.getframerate())
rec.SetWords(True)  # Enable the result to include the words
rec.SetPartialWords(True)  # Enable the result to include the partial words

text = ""  # The text of the audio
words = []  # The tokenized words of the audio
start_times = []  # The start times of the spoken words
end_times = []  # The end times of the spoken words

data = wf.readframes(wf.getnframes())  # Read the audio file in chunks of 4000 frames
if len(data) > 0:
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

        output_folder_trimmed = "trimmed"
        trimming.trim_on_descending_waveform(file_path, start_times, end_times, words, number_of_words,
                                             output_folder_trimmed, True)

        print("\nTRANSCRIBED TEXT")
        print(text)
        print("TOKENIZED WORDS")
        print(words)
        print("WORDS START TIMES")
        print(start_times)
        print("WORDS END TIMES")
        print(end_times)

        # Extract the original file name without extension
        original_file_name = os.path.splitext(os.path.basename(file_path))[0]

        # Print the location of the trimmed segments
        print(f"\nTrimming completed. Files stored into: {output_folder_trimmed}/{original_file_name}_trimmed*.wav")