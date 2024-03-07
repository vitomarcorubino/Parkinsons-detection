# Source: https://huggingface.co/speechbrain/lang-id-commonlanguage_ecapa
from speechbrain.inference.classifiers import EncoderClassifier

def get_language_id(file_path):
    language_id = EncoderClassifier.from_hparams(source="TalTechNLP/voxlingua107-epaca-tdnn", savedir="tmp")
    signal = language_id.load_audio(file_path)
    prediction = language_id.classify_batch(signal)
    lang_id = prediction[-1][0]
    return lang_id
