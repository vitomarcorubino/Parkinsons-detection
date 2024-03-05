from speechbrain.inference.ASR import EncoderDecoderASR

# Load the ASR model
asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-commonvoice-it", savedir="pretrained_models/asr-crdnn-commonvoice-it")

# Transcribe the audio file
audio_file = 'audio/italiano.wav'
transcription = asr_model.transcribe_file(audio_file)

# Print the transcription
print(f"Transcription: {transcription}")
