# 🔍 CNN and Attention Mechanisms for Parkinson's Diagnosis and Speech Deficit Detection
This repository contains the implementation of a research project aimed at analyzing audio recordings using a deep learning approach involving a Convolutional Neural Network.

## 📊 Interpretability
To enhance the interpretability of the model, an Attention Grad-CAM mechanism was implemented. This technique generates heatmaps that highlight the most relevant segments of the audio recordings for classification purposes. 

These two heatmaps show what Attention Mechanism produced for a recording of a healthy patient and for a recording of a patient with Parkinson's disease. <br> <br>
<img src="https://github.com/vitomarcorubino/Parkinsons-detection/assets/98357718/91aa62d8-dea9-40b9-b370-25c6be371eee" height="300" /> <br>
<img src="https://github.com/vitomarcorubino/Parkinsons-detection/assets/98357718/98d2e3ba-9a26-4a9f-baff-8b7c201dc66d" height="300" /> <br>

## 🔬 Lexical Analysis
From the significant segments identified by the Attention Grad-CAM, the Vosk toolkit was used to transcribe the audio.
A lexical analysis was then performed to identify the frequency of crucial words in diagnosing Parkinson's disease. 
This analysis facilitated the creation of a new text, optimized for future recording sessions, which aims to streamline the data collection process while reducing the vocal strain on patients.

## 📚 Bachelor's thesis
The bachelor's thesis associated with this repository can be found at this link: <br> <br>
[![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/file/d/1i3Mt7KYvnbZ4b4gdz-SoJx6Bo-aomNzi/view?usp=sharing) 

The thesis contains detailed information about feature extraction, the implementation of the Deep Learning model, and the Grad-CAM Attention mechanism.
