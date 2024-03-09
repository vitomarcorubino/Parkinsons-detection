from pydub import AudioSegment
import os


# check if the audio file is in mono PCM format
def is_mono_pcm(wav_file):
    if wav_file.getnchannels() == 1 and wav_file.getsampwidth() == 2 and wav_file.getcomptype() == "NONE":
        return True
    else:
        return False


# convert the audio file to mono PCM format
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
