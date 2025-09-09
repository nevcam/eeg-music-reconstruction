# Reconstructing Sound From Brain Responses To Music Stimuli

## Abstract
This study explores reconstructing music stimuli from EEG signals, building on the EEG2Mel paper. Using the NMED-T dataset, we preprocessed EEG data into power spectral density representations and paired it with mel-spectrogram outputs of the music stimuli. We implemented different models architectures to map EEG signals to audio features, optimizing performance through rigorous preprocessing and model tuning. Our best model outperformed the baseline EEG2Mel study with accuracy of 82.86% (cosine similarity: 0.80). This research highlights the potential of deep learning in decoding brain responses to music and provides insights for future advancements in neural signal processing.

## Dataset
We used the Naturalistic Music EEG Dataset – Tempo (NMED-T), which contains EEG recordings of 20 participants listening to 10 full-length songs (4:30–5:00 min).
* **Sampling rate:** 1 kHz (downsampled to 125 Hz)
* **Preprocessing:** High-pass and notch filtering, ICA artifact removal, aggregation
* **Final dataset size:** 54,000 EEG–audio pairs

## Preprocessing

EEG: Trimmed to 270s, segmented into 1-second chunks → PSD computed using Welch’s method and Periodogram.
Audio: Converted into mel-spectrograms in both raw and decibel (dB) scale formats.
Final dimensions:
* **EEG (X)**: (54000, 63, 125, 1)
* **Mel-spectrogram (Y)**: (54000, 128, 44)

## Model Architectures

* **Baseline CNN**: Replicated EEG2Mel’s approach for benchmarking.
* **RCNN**: Combined convolutional layers with recurrent layers to capture temporal dependencies.
* **Encoder–Decoder**: LSTM-based sequence modeling to reconstruct mel-spectrograms.
* **Transformers**: Explored self-attention for long-range dependencies.

## Implementation
* **Frameworks:** PyTorch, TensorFlow, Keras
* **Audio Processing:** Librosa
* **Metrics:** Cosine similarity, EEG song classifier (repurposed from EEG2Mel for comparability)
* **Compute:** Google Colab Pro (NVIDIA T4 GPUs, ~215 hrs total training)

## References
[1] Ramirez-Aristizabal, Adolfo G. and Christopher T. Kello. “EEG2Mel: Reconstructing Sound from Brain Responses to Music.” ArXiv abs/2207.13845 (2022): n. pag.

[2] Steven Losorelli, Duc T. Nguyen, Jacek P. Dmochowski, and Blair Kaneshiro (2017). Naturalistic Music EEG Dataset - Tempo (NMED-T). Stanford Digital Repository.

[3] Steven Losorelli, Duc T. Nguyen, Jacek P. Dmochowski, and Blair Kaneshiro (to appear). NMED-T: A Tempo-Focused Dataset of Cortical and Behavioral Responses to Naturalistic Music. In Proceedings of the 18th International Society for Music Information Retrieval Conference, Suzhou, China.

[4] M.-A. Moinnereau, T. Brienne, S. Brodeur, J. Rouat, K. Whittingstall, and E. Plourde, “Classification of auditory stimuli from eeg signals with a regulated recurrent neural network reservoir,” arXiv preprint arXiv:1804.10322, 2018

[5] A. G. Ramirez-Aristizabal, M. K. Ebrahimpour, and C. T. Kello, “Image-based eeg classification of brain responses to song recordings,” arXiv preprint arXiv:2202.03265, 2022.

[6] Zhang, B., Leitner, J., Thornton, S. (2019). Audio Recognition using Mel Spectrograms and Convolution Neural Networks.

[7] Vaswani, Ashish et al. “Attention is All you Need.” Neural Information Processing Systems (2017).
