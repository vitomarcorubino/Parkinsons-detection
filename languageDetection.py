# Source: https://huggingface.co/speechbrain/lang-id-commonlanguage_ecapa
from speechbrain.inference.classifiers import EncoderClassifier
import os

def get_language_id(file_path):
    """
    This function is used to identify the language of an audio file and return the language id.

    Args:
        file_path (str): The path to the audio file that needs to be classified.

    Returns:
        str: The identified language id of the audio file.
    """
    language_id = EncoderClassifier.from_hparams(source="TalTechNLP/voxlingua107-epaca-tdnn", savedir="tmp")
    savedir = os.path.dirname(file_path)
    signal = language_id.load_audio(file_path, savedir=savedir)
    prediction = language_id.classify_batch(signal)
    lang_id = prediction[-1][0]

    supported_languages = get_supported_languages()

    if lang_id not in supported_languages:
        lang_id = "en"

    return lang_id


def get_supported_languages():
    """
    This function returns a list of supported languages for the language detection process.
    The languages are represented by their respective language codes.

    Returns:
        list: A list of language codes representing the supported languages.
    """
    supported_languages = [
        "en", "en-us", "en-in", "cn", "ru", "fr", "de", "es", "pt", "gr", "tr",
        "vn", "it", "nl", "ca", "ar", "fa", "tl-ph", "uk", "kz", "sv", "eo",
        "hi", "pl", "uz", "ko", "br", "gu"
    ]

    return supported_languages
