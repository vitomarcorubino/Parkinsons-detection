import glob
import os
from datetime import datetime

import natsort
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call


class FeatureExtraction:
    """
    Feature extraction class containing the methods to extract features for each voice sample

    Attributes:
    acoustic_features : list
         a list of acoustic features such as jitters and shimmers for a voice sample
    mfcc: list
         a list of mfcc extracted from the voice sample

    """

    def __init__(self):
        self.acoustic_features = []
        self.mfcc = []

    def extract_acoustic_features(self, voice_sample, f0_min, f0_max, unit):
        """
        Extracts the acoustic features such as the jitters and shimmers from the voice sample using functions from Praat software

        Parameters:
        voice_sample : .wav file
            the voice sample we want to extract the features from
        f0_min: integer
            minimum fundamental frequency of the signal. defualt is 75 on Praat software
        f0_max: integer
            maximum fundamental frequency of the signal. default is 500 on Praat software

        """
        try:
            sound = parselmouth.Sound(voice_sample)
            pitch = call(sound, "To Pitch", 0.0, f0_min, f0_max)
            f0_mean = call(pitch, "Get mean", 0, 0, unit)
            f0_std_deviation = call(pitch, "Get standard deviation", 0, 0, unit)
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0_min, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
            pointProcess = call(sound, "To PointProcess (periodic, cc)", f0_min, f0_max)
            jitter_relative = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_absolute = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_rap = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_ppq5 = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer_relative = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_localDb = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq3 = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq5 = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

            return f0_mean, f0_std_deviation, hnr, jitter_relative, jitter_absolute, jitter_rap, jitter_ppq5, shimmer_relative, shimmer_localDb, shimmer_apq3, shimmer_apq5
        except:
            print("Unable to process this file: ", voice_sample)

    def extract_acoustic_features_2(self, voice_sample, f0_min, f0_max, unit):
        """
        Extracts the acoustic features such as the jitters and shimmers from the voice sample using functions from Praat software

        These fetaures are the features used in the research paper that used mdvr_kcl dataset

        Parameters:
        voice_sample : .wav file
            the voice sample we want to extract the features from
        f0_min: integer
            minimum fundamental frequency of the signal. defualt is 75 on Praat software
        f0_max: integer
            maximum fundamental frequency of the signal. default is 500 on Praat software

        """
        try:
            sound = parselmouth.Sound(voice_sample)
            pitch = call(sound, "To Pitch", 0.0, f0_min, f0_max)
            f0_mean = call(pitch, "Get mean", 0, 0, unit)
            f0_max = call(pitch, "Get maximum", 0, 0, unit, "Parabolic")
            f0_min = call(pitch, "Get minimum", 0, 0, unit, "Parabolic")
            f0_std_deviation = call(pitch, "Get standard deviation", 0, 0, unit)
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0_min, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
            pointProcess = call(sound, "To PointProcess (periodic, cc)", f0_min, f0_max)
            jitter_relative = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_absolute = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_rap = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_ddp = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_ppq5 = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer_relative = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_localDb = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq3 = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq5 = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_dda = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

            return f0_mean, f0_max, f0_min, jitter_relative, jitter_absolute, jitter_rap, jitter_ddp, shimmer_relative, shimmer_localDb, shimmer_apq3, shimmer_apq5, shimmer_dda, hnr
        except Exception as e:
            print("Unable to process this file: ", voice_sample)
            print(e)
            return (None,) * 13  # return a tuple of None values

    def extract_mfcc(self, voice_sample):
        """
        Extracts the mel frequency ceptral coefficients from the voice sample

        Parameters:
        voice_sample : .wav file
            the voice sample we want to extract the features from
        """

        sound = parselmouth.Sound(voice_sample)
        mfcc_object = sound.to_mfcc(number_of_coefficients=12)  #the optimal number of coeefficient used is 12
        mfcc = mfcc_object.to_array()
        mfcc_mean = np.mean(mfcc.T, axis=0)
        return mfcc_mean

    def extract_mfcc_from_folder(self, folder_path):
        file_list = []
        mfcc_list = []
        features = []
        curr_time = datetime.now()
        print("Entering extract_mfcc_from_folder, time:", curr_time)
        for file in glob.glob(folder_path):
            try:
                #print("Processing file:", file)
                mfcc_per_file = self.extract_mfcc(file)
                #mfcc_list.append(mfcc_for_file)
                #file_list.append(file)
                features.append([file, mfcc_per_file])
            except:
                print("error while handling file: ", file)
        #df = pd.DataFrame(file_list, mfcc_list)
        df = pd.DataFrame(features, columns=['voiceID', 'mfcc'])
        df[['mfcc_feature0', 'mfcc_feature1', 'mfcc_feature2', 'mfcc_feature3', 'mfcc_feature4', 'mfcc_feature5',
            'mfcc_feature6', 'mfcc_feature7', 'mfcc_feature8', 'mfcc_feature9', 'mfcc_feature10', 'mfcc_feature11',
            'mfcc_feature12']] = pd.DataFrame(df.mfcc.to_list())
        df = df.drop(columns=['mfcc'])
        return df

    def extract_features(self, wav_filepath):
        df_dict = {}

        # Create column names for each MFCC
        mfcc_columns = [f'mfcc_{i}' for i in range(13)]

        # Combine these with the existing column names
        columns = ['meanF0Hz', 'stdevF0Hz', 'HNR', 'localJitter', 'localabsoluteJitter', 'rapJitter',
                   'ppq5Jitter', 'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer'] + mfcc_columns

        # print("Processing:", wav_filepath)

        # Initialize the lists to store the features
        filename_list = []
        mean_F0_list = []
        sd_F0_list = []
        hnr_list = []
        localJitter_list = []
        localabsoluteJitter_list = []
        rapJitter_list = []
        ppq5Jitter_list = []
        localShimmer_list = []
        localdbShimmer_list = []
        apq3Shimmer_list = []
        aqpq5Shimmer_list = []
        mfcc_list = []

        # Get the base name of the wav file
        base_name = os.path.splitext(os.path.basename(wav_filepath))[0]

        # Get the directory of the wav file
        dir_name = os.path.dirname(wav_filepath)

        # Get the last folder of the wav file
        last_folder = os.path.basename(dir_name)

        if last_folder == "mono_pcm":
            # Get the parent directory of the wav file
            parent_dir_name = os.path.dirname(dir_name)

            # Get all the trimmed files corresponding to the wav file
            trimmed_files = glob.glob(os.path.join(parent_dir_name, "trimmed", base_name + "_trimmed*.wav"))
        else:
            # Get all the trimmed files corresponding to the wav file
            trimmed_files = glob.glob(os.path.join(dir_name, "trimmed", base_name + "_trimmed*.wav"))

        trimmed_files = natsort.natsorted(trimmed_files)

        # Loop over the trimmed files
        for trimmed_file in trimmed_files:
            # Extract the features
            (meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, localShimmer, localdbShimmer,
            apq3Shimmer, aqpq5Shimmer) = self.extract_acoustic_features(trimmed_file, 75, 500, "Hertz")

            # Append the features to the lists
            filename_list.append(trimmed_file)
            mean_F0_list.append(meanF0)
            sd_F0_list.append(stdevF0)
            hnr_list.append(hnr)
            localJitter_list.append(localJitter)
            localabsoluteJitter_list.append(localabsoluteJitter)
            rapJitter_list.append(rapJitter)
            ppq5Jitter_list.append(ppq5Jitter)
            localShimmer_list.append(localShimmer)
            localdbShimmer_list.append(localdbShimmer)
            apq3Shimmer_list.append(apq3Shimmer)
            aqpq5Shimmer_list.append(aqpq5Shimmer)
            mfcc_list.append(self.extract_mfcc(trimmed_file))

        # Create a DataFrame for the .wav file with its corresponding trimmed files' features
        df = pd.DataFrame(np.column_stack(
            [mean_F0_list, sd_F0_list, hnr_list, localJitter_list, localabsoluteJitter_list,
            rapJitter_list, ppq5Jitter_list, localShimmer_list, localdbShimmer_list, apq3Shimmer_list,
            aqpq5Shimmer_list, mfcc_list]), columns=columns)

        df = df.transpose()

        df_dict[base_name] = df.to_numpy()

        return df_dict

    def extract_features_from_folder(self, main_folder_path):
        # df_dict_list = []
        # labels_list = []  # list to store the labels
        labels = []
        df_dict = {}

        # Create column names for each MFCC
        mfcc_columns = [f'mfcc_{i}' for i in range(13)]

        # Combine these with the existing column names
        columns = ['meanF0Hz', 'stdevF0Hz', 'HNR', 'localJitter', 'localabsoluteJitter', 'rapJitter',
                   'ppq5Jitter', 'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer'] + mfcc_columns

        # wav_file_counter = 0
        # Store the original main_folder_path
        original_main_folder_path = main_folder_path
        # Loop over the main folders
        for main_folder in os.listdir(original_main_folder_path):
            # df_dict = {}  # dictionary to store the dataframes
            # labels = []
            main_folder_path = os.path.join(original_main_folder_path, main_folder)

            print("Processing:", main_folder_path)
            # Get the list of .wav files
            wav_files = glob.glob(os.path.join(main_folder_path, "*.wav"))

            # Loop over the .wav files
            for wav_file in wav_files:
                # wav_file_counter += 1

                # Skip the first 642 .wav files
                # if wav_file_counter <= 642:
                #    continue

                if main_folder == "peopleWithParkinson":
                    labels.append(1)
                else:
                    labels.append(0)

                # Initialize the lists to store the features
                filename_list = []
                mean_F0_list = []
                sd_F0_list = []
                hnr_list = []
                localJitter_list = []
                localabsoluteJitter_list = []
                rapJitter_list = []
                ppq5Jitter_list = []
                localShimmer_list = []
                localdbShimmer_list = []
                apq3Shimmer_list = []
                aqpq5Shimmer_list = []
                mfcc_list = []

                # Get the corresponding trimmed files
                trimmed_files = glob.glob(os.path.join(main_folder_path, "trimmed", os.path.splitext(os.path.basename(wav_file))[0] + "_trimmed*.wav"))
                trimmed_files = natsort.natsorted(trimmed_files)

                # Loop over the trimmed files
                for trimmed_file in trimmed_files:
                    # Extract the features
                    (meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, localShimmer, localdbShimmer,
                    apq3Shimmer, aqpq5Shimmer) = self.extract_acoustic_features(trimmed_file, 75, 500, "Hertz")

                    # Append the features to the lists
                    filename_list.append(trimmed_file)
                    mean_F0_list.append(meanF0)
                    sd_F0_list.append(stdevF0)
                    hnr_list.append(hnr)
                    localJitter_list.append(localJitter)
                    localabsoluteJitter_list.append(localabsoluteJitter)
                    rapJitter_list.append(rapJitter)
                    ppq5Jitter_list.append(ppq5Jitter)
                    localShimmer_list.append(localShimmer)
                    localdbShimmer_list.append(localdbShimmer)
                    apq3Shimmer_list.append(apq3Shimmer)
                    aqpq5Shimmer_list.append(aqpq5Shimmer)
                    mfcc_list.append(self.extract_mfcc(trimmed_file))

                # Create a DataFrame for the .wav file with its corresponding trimmed files' features
                df = pd.DataFrame(np.column_stack(
                    [mean_F0_list, sd_F0_list, hnr_list, localJitter_list, localabsoluteJitter_list,
                    rapJitter_list, ppq5Jitter_list, localShimmer_list, localdbShimmer_list, apq3Shimmer_list,
                    aqpq5Shimmer_list, mfcc_list]), columns=columns)

                # df = df.transpose()

                df_dict[os.path.basename(wav_file)] = df.to_numpy()

            # df_dict_list.append(df_dict)
            # labels_list.append(labels)

        return df_dict, labels

    def extract_features_from_folder_2(self, folder_path):  #for the MDVR_KCL dataset (replication)
        df = pd.DataFrame()
        file_list = []
        mean_F0_list = []
        max_F0_list = []
        min_F0_list = []
        hnr_list = []
        localJitter_list = []
        localabsoluteJitter_list = []
        rapJitter_list = []
        ddpJitter_list = []
        localShimmer_list = []
        localdbShimmer_list = []
        apq3Shimmer_list = []
        aqpq5Shimmer_list = []
        ddaShimmer_list = []
        curr_time = datetime.now()
        print("Entering extract_features_from_folder_2, time:", curr_time)
        for file in glob.glob(folder_path):
            print("extract_features_from_folder_2", file)
            (meanF0, maxF0, minF0, localJitter, localabsoluteJitter, rapJitter, ddpJitter, localShimmer, localdbShimmer,
             apq3Shimmer, aqpq5Shimmer, ddaShimmer, hnr) = self.extract_acoustic_features_2(file, 75, 500, "Hertz")

            # Check if a None tuple is returned
            if (meanF0,) != (None,):
                file_list.append(file)  # make an ID list
                mean_F0_list.append(meanF0)  # make a mean F0 list
                max_F0_list.append(maxF0)
                min_F0_list.append(minF0)
                localJitter_list.append(localJitter)
                localabsoluteJitter_list.append(localabsoluteJitter)
                rapJitter_list.append(rapJitter)
                ddpJitter_list.append(ddpJitter)
                localShimmer_list.append(localShimmer)
                localdbShimmer_list.append(localdbShimmer)
                apq3Shimmer_list.append(apq3Shimmer)
                aqpq5Shimmer_list.append(aqpq5Shimmer)
                ddaShimmer_list.append(ddaShimmer)
                hnr_list.append(hnr)
            else:
                print("Missed:", file)

        df = pd.DataFrame(np.column_stack(
            [file_list, mean_F0_list, max_F0_list, min_F0_list, localJitter_list, localabsoluteJitter_list,
             rapJitter_list, ddpJitter_list, localShimmer_list, localdbShimmer_list, apq3Shimmer_list,
             aqpq5Shimmer_list, ddaShimmer_list, hnr_list]),
                          columns=['voiceID', 'meanF0Hz', 'maxF0Hz', 'minF0Hz', 'localJitter', 'localabsoluteJitter',
                                   'rapJitter', 'ddpJitter', 'localShimmer', 'localdbShimmer', 'apq3Shimmer',
                                   'apq5Shimmer', 'ddaShimmer', 'hnr'])
        return df

    def extract_features_from_folder_3(self, main_folder_path):
        # df_dict_list = []
        # labels_list = []  # list to store the labels
        labels = []
        df_dict = {}
        df = pd.DataFrame()

        # Create column names for each MFCC
        mfcc_columns = [f'mfcc_{i}' for i in range(13)]

        # Combine these with the existing column names
        columns = ['meanF0Hz', 'stdevF0Hz', 'HNR', 'localJitter', 'localabsoluteJitter', 'rapJitter',
                   'ppq5Jitter', 'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer'] + mfcc_columns

        # Initialize the lists to store the features
        filename_list = []
        mean_F0_list = []
        sd_F0_list = []
        hnr_list = []
        localJitter_list = []
        localabsoluteJitter_list = []
        rapJitter_list = []
        ppq5Jitter_list = []
        localShimmer_list = []
        localdbShimmer_list = []
        apq3Shimmer_list = []
        aqpq5Shimmer_list = []
        mfcc_list = []
        # wav_file_counter = 0
        # Store the original main_folder_path
        original_main_folder_path = main_folder_path
        # Loop over the main folders
        for trimmed_file in glob.glob(original_main_folder_path):
            # df_dict = {}  # dictionary to store the dataframes
            # labels = []

            print("Processing:", trimmed_file)

            # Extract the features
            (meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, localShimmer, localdbShimmer,
            apq3Shimmer, aqpq5Shimmer) = self.extract_acoustic_features(trimmed_file, 75, 500, "Hertz")

            # Append the features to the lists
            filename_list.append(trimmed_file)
            mean_F0_list.append(meanF0)
            sd_F0_list.append(stdevF0)
            hnr_list.append(hnr)
            localJitter_list.append(localJitter)
            localabsoluteJitter_list.append(localabsoluteJitter)
            rapJitter_list.append(rapJitter)
            ppq5Jitter_list.append(ppq5Jitter)
            localShimmer_list.append(localShimmer)
            localdbShimmer_list.append(localdbShimmer)
            apq3Shimmer_list.append(apq3Shimmer)
            aqpq5Shimmer_list.append(aqpq5Shimmer)
            mfcc_list.append(self.extract_mfcc(trimmed_file))

            # Create a DataFrame for the .wav file with its corresponding trimmed files' features
            df = pd.DataFrame(np.column_stack(
                [mean_F0_list, sd_F0_list, hnr_list, localJitter_list, localabsoluteJitter_list,
                rapJitter_list, ppq5Jitter_list, localShimmer_list, localdbShimmer_list, apq3Shimmer_list,
                aqpq5Shimmer_list, mfcc_list]), columns=columns)

            df = df.transpose()

            df_dict[os.path.basename(trimmed_file)] = df.to_numpy()

        # df_dict_list.append(df_dict)
        # labels_list.append(labels)

        return df

    def extract_features_from_folder4(self, main_folder_path, trimmed = False):
        # df_dict_list = []
        # labels_list = []  # list to store the labels
        labels = []
        df_dict = {}

        # Create column names for each MFCC
        mfcc_columns = [f'mfcc_{i}' for i in range(13)]

        # Combine these with the existing column names
        columns = ['meanF0Hz', 'stdevF0Hz', 'HNR', 'localJitter', 'localabsoluteJitter', 'rapJitter',
                   'ppq5Jitter', 'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer'] + mfcc_columns

        # wav_file_counter = 0
        # Store the original main_folder_path
        original_main_folder_path = main_folder_path
        # Loop over the main folders
        for main_folder in os.listdir(original_main_folder_path):
            # df_dict = {}  # dictionary to store the dataframes
            # labels = []
            main_folder_path = os.path.join(original_main_folder_path, main_folder)

            print("Processing:", main_folder_path)

            for patient_folder in os.listdir(main_folder_path):

                patient_folder_path = os.path.join(main_folder_path, patient_folder)
                # Get the list of .wav files
                wav_files = glob.glob(os.path.join(patient_folder_path, "*.wav"))

                # Loop over the .wav files
                for wav_file in wav_files:
                    # wav_file_counter += 1

                    # Skip the first 642 .wav files
                    # if wav_file_counter <= 642:
                    #    continue

                    if main_folder == "peopleWithParkinson":
                        labels.append(1)
                    else:
                        labels.append(0)

                    # Initialize the lists to store the features
                    filename_list = []
                    mean_F0_list = []
                    sd_F0_list = []
                    hnr_list = []
                    localJitter_list = []
                    localabsoluteJitter_list = []
                    rapJitter_list = []
                    ppq5Jitter_list = []
                    localShimmer_list = []
                    localdbShimmer_list = []
                    apq3Shimmer_list = []
                    aqpq5Shimmer_list = []
                    mfcc_list = []

                    # Get the corresponding trimmed files
                    trimmed_files = glob.glob(os.path.join(patient_folder_path, "trimmed", os.path.splitext(os.path.basename(wav_file))[0] + "_trimmed*.wav"))
                    trimmed_files = natsort.natsorted(trimmed_files)

                    # Loop over the trimmed files
                    for trimmed_file in trimmed_files:
                        # Extract the features
                        (meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, localShimmer, localdbShimmer,
                        apq3Shimmer, aqpq5Shimmer) = self.extract_acoustic_features(trimmed_file, 75, 500, "Hertz")

                        # Append the features to the lists
                        filename_list.append(trimmed_file)
                        mean_F0_list.append(meanF0)
                        sd_F0_list.append(stdevF0)
                        hnr_list.append(hnr)
                        localJitter_list.append(localJitter)
                        localabsoluteJitter_list.append(localabsoluteJitter)
                        rapJitter_list.append(rapJitter)
                        ppq5Jitter_list.append(ppq5Jitter)
                        localShimmer_list.append(localShimmer)
                        localdbShimmer_list.append(localdbShimmer)
                        apq3Shimmer_list.append(apq3Shimmer)
                        aqpq5Shimmer_list.append(aqpq5Shimmer)
                        mfcc_list.append(self.extract_mfcc(trimmed_file))

                    # Create a DataFrame for the .wav file with its corresponding trimmed files' features
                    df = pd.DataFrame(np.column_stack(
                        [mean_F0_list, sd_F0_list, hnr_list, localJitter_list, localabsoluteJitter_list,
                        rapJitter_list, ppq5Jitter_list, localShimmer_list, localdbShimmer_list, apq3Shimmer_list,
                        aqpq5Shimmer_list, mfcc_list]), columns=columns)

                    df = df.transpose()

                    df_dict[os.path.basename(wav_file)] = df.to_numpy()

                # df_dict_list.append(df_dict)
                # labels_list.append(labels)

        return df_dict, labels

    def convert_to_csv(self, df, filename):
        df.to_csv(filename + ".csv", index=False)
