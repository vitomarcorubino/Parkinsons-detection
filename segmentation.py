# Source: https://stackoverflow.com/a/36461422
from pydub import AudioSegment
from pydub.silence import split_on_silence

# Load the audio file
sound_file = AudioSegment.from_wav("audio/italiano.wav")
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
    out_file = "audio/chunk{0}.wav".format(i)
    print("exporting", out_file)
    chunk.export(out_file, format="wav")

# LSTM o modelli deep che lavorino su serie temporali