# Source: https://huggingface.co/speechbrain/lang-id-commonlanguage_ecapa
from speechbrain.inference.classifiers import EncoderClassifier


def get_language_id(file_path):
    """
    This function is used to identify the language of an audio file and return the language id.

    Args:
        file_path (str): The path to the audio file that needs to be classified.

    Returns:
        str: The identified language id of the audio file.
    """
    language_id = EncoderClassifier.from_hparams(source="TalTechNLP/voxlingua107-epaca-tdnn", savedir="tmp")
    signal = language_id.load_audio(file_path)
    prediction = language_id.classify_batch(signal)
    lang_id = prediction[-1][0]
    return lang_id
