# Source: https://github.com/alphacep/vosk-api/blob/master/python/example/test_simple.py
import os
import json
import wave
import audioFileFormat  # Module to handle audio file format and conversion to wav mono pcm
import languageDetection  # Module to detect the language of the audio file
import trimming  # Module to trim the audio file
from vosk import Model, KaldiRecognizer, SetLogLevel
import shutil

audio_dir = "newDataset/elderlyHealthyControl/GiovannaAnaclerio"  # The directory where the audio files are stored
output_folder_converted = "newDataset/elderlyHealthyControl/GiovannaAnaclerio/mono_pcm"  # The folder where the converted audio are stored
output_folder_trimmed = "newDataset/elderlyHealthyControl/GiovannaAnaclerio/trimmed"  # The folder where the trimmed audio files are stored
number_of_words = 4  # The number of words to consider for each trimming operation

# Check if audio_dir is a directory or a file
if os.path.isdir(audio_dir):
    # If it's a directory, get the list of audio files in the audio directory
    wav_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
else:
    if os.path.isfile(audio_dir):
        # If it's a file, create a list with the file
        wav_files = [audio_dir]
    else:
        print("The specified path is not a valid file or directory.")
        wav_files = []

# -1 to disable logging messages, 0 to enable them
SetLogLevel(-1)

# Iterate over the audio files
for wav_file in wav_files:
    # Construct the full file path
    if os.path.isdir(audio_dir):
        file_path = os.path.join(audio_dir, wav_file)
    else:
        file_path = audio_dir

    # Open the audio file
    wf = wave.open(file_path, "rb")
    if not audioFileFormat.is_mono_pcm(wf):
        print(f"\nAudio file {wav_file} must be WAV format mono PCM.")
        print("Converting to mono PCM...")

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

            if 'result' in result_json:
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

                trimming.trim_on_descending_waveform(file_path, start_times, end_times, words, number_of_words,
                                                     output_folder_trimmed, False)

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
            else:
                print("No result found. Copying the input file to the output directory.")

                # Extract the original file name without extension
                original_file_name = os.path.splitext(os.path.basename(file_path))[0]

                # Construct the output file path
                output_file_path = os.path.join(output_folder_trimmed, f"{original_file_name}_trimmed0.wav")

                # Copy the input file to the output directory
                shutil.copy2(file_path, output_file_path)

                print(f"File copied to: {output_file_path}")
