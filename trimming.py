from scipy.io.wavfile import read, write
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
import plotting


def trim_on_descending_waveform(audio_path, start_times, end_times, words, number_of_words, threshold=0.1,
                                end_buffer=0.075):
    """
    Trims an audio file based on the start and end times of spoken words. It iterates over the start and end times,
    extracts the corresponding audio segment, and trims the segment at the first point where the amplitude decreases.
    The trimmed segments are then exported as separate audio files.

    Args:
        audio_path (str): The path to the audio file to be trimmed.
        start_times (list): A list of start times for each spoken word in the audio file.
        end_times (list): A list of end times for each spoken word in the audio file.
        words (list): A list of the spoken words in the audio file.
        number_of_words (int): The number of words to consider for each trimming operation.
        threshold (float, optional): The threshold for determining a decrease in amplitude. Defaults to 0.1.
        end_buffer (float, optional): The buffer to add to the end time of each segment, in seconds. Defaults to 0.075.
    """

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
            end_sample = int((end_times[-1]) * sample_rate + end_buffer * sample_rate)
        else:
            end_sample = int((end_times[i + number_of_words - 1]) * sample_rate + end_buffer * sample_rate)

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

    plotting.plot_trimmed_audio(audio_path, start_times, end_times, 4, words)


def trim_on_timestamp(audio_path, start_times, end_times, number_of_words):
    """
    Trims an audio file into segments based on the start and end times of spoken words.
    Each segment is then exported as a separate audio file.

    Args:
        audio_path (str): Path to the audio file.
        start_times (list): Start times of each spoken word.
        end_times (list): End times of each spoken word.
        number_of_words (int): Number of words to consider for each slicing operation.
    """
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
        sliced_audio.export(f"audio/trimmed_audio{i // number_of_words}.wav", format="wav")

        i = i + number_of_words


def trim_on_silence(audio_file_path):
    """
    Trims an audio file into chunks based on silence. It uses the 'split_on_silence' function from the 'pydub.silence' module to detect silent parts of the audio and split the audio at these points. Each chunk is then exported as a separate audio file.

    Args:
        audio_file_path (str): Path to the audio file to be split.
    """

    # Load the audio file
    sound_file = AudioSegment.from_wav(audio_file_path)

    # Split track and get chunks
    audio_chunks = split_on_silence(sound_file,
                                    # must be silent for at least 180 ms
                                    min_silence_len=180,

                                    # consider it silent if quieter than -30 dBFS
                                    silence_thresh=-30,

                                    # keep 100 ms of leading/trailing silence
                                    keep_silence=150
                                    )

    # Print number of chunks
    print(f"Number of chunks created: {len(audio_chunks)}")

    # Export each chunk as a wav file
    for i, chunk in enumerate(audio_chunks):
        out_file = "audio/trimmed_audio{0}.wav".format(i)
        print("exporting", out_file)
        chunk.export(out_file, format="wav")
