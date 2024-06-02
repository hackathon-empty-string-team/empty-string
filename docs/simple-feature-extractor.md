## What happens in the simple feature extractor?

A high level step by step explanation what happens to a single soundfile: 

1. read a given sound file (currently only .wav supported)
2. create a spectrogram for the given file 
3. set time windows (to be set in the frontend or in the cockpit (notebook))
4. set frequency windows (to be set in the frontend or in the cockpit (notebook))
5. split spectrogram into "patches"
6. extract individual spectral components for said patches using [FFT](https://librosa.org/doc/0.10.2/generated/librosa.fft_frequencies.html#librosa-fft-frequencies)
7. extract features for the patches using a single [2D-convolutional](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) layer using [GlorotUniform (Xavier)](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) initialization with a fixed random seed
8. gather and return the features, along with the corresponding time windows and filgather and return the features, along with the corresponding time windows and filename


