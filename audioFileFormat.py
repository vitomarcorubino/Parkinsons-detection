from pydub import AudioSegment
import os


def is_mono_pcm(wav_file):
    """
    This function checks if a given audio file is in mono PCM format. It checks three properties of the audio file:
    1. The number of channels: It should be 1 for mono audio.
    2. The sample width: It should be 2 for PCM format.
    3. The compression type: It should be "NONE" for PCM format.

    Args:
        wav_file (wave.Wave_read): The audio file that needs to be checked.

    Returns:
        bool: True if the audio file is in mono PCM format, False otherwise.
    """
    if wav_file.getnchannels() == 1 and wav_file.getsampwidth() == 2 and wav_file.getcomptype() == "NONE":
        return True
    else:
        return False


# convert the audio file to mono PCM format
def convert_to_mono_pcm(wav_file_path):
    """
    This function converts a given audio file to mono PCM format. It performs two operations on the audio file:
    1. Converts the audio to mono by setting the number of channels to 1.
    2. Sets the sample width to 2, which is characteristic of PCM format.

    The function then exports the converted audio as a new WAV file with '_mono_pcm' appended to the original file name.

    Args:
        wav_file_path (str): The path to the audio file that needs to be converted.
    """
    audio = AudioSegment.from_wav(wav_file_path)

    # Convert to mono
    mono_audio = audio.set_channels(1)

    # Set sample width to 2
    mono_audio = mono_audio.set_sample_width(2)

    # Get the original file name without extension
    original_file_name = os.path.basename(os.path.splitext(wav_file_path)[0])

    # Create the output file name
    output_file_name = f"audio/{original_file_name}_mono_pcm.wav"

    # Export as PCM WAV
    mono_audio.export(output_file_name, format="wav")
