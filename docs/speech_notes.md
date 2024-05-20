#Notes from a discussion with a speech PhD student 

#Processing Pipeline for Human Speech Recordings

Imagine you have 10-minute recordings of human speech (5 minutes of speech, 5 minutes of background noise). The typical processing pipeline involves:

1. **Voice Activity Detection** (removing non-speech parts)
2. **Feature Extraction** (from relevant parts)
3. **Classification**

You can either use specific models for each step, trained individually or pre-trained, or a model that optimizes all three steps together with a combined loss function for potentially better results. The main focus is on feature extraction, comparing spectral/handcrafted features like **MFCCs**, **vanilla spectrograms**, **Mel-Spectrograms**, and low-level descriptors such as **eGeMaps** and **openSMILE** with embeddings from state-of-the-art deep learning models. These features are aggregated into vectors for classification, typically using an **MLP/transformer model**, to determine the best performance with the chosen metrics. Using a model like **WavLM**, you can test all layers to find the most effective one for the task.


# Processing Pipeline for Animal Sounds

Now, for animal sounds rather than human sounds, things get a bit more complicated. **MFCCs** and even other models like **HuBERT** are all biased towards humans. As soon as the input features are on a **Mel scale** or **log-mel**, they are biased towards human perception and frequencies because the production of vocalizations in any animal (including humans) is fundamentally linked to the perception of vocalizations.

In this case, you can:

- Transform the frequency scale of your spectrogram to amplify the vocalizations, such as using a **log-spectrogram** or a **bark-spectrogram**.
- Use **GFCCs** (Greenwood Function Cepstral Coefficients) instead of MFCCs, which are less biased towards humans.
- Use **time-series** and **signal processing features**, and stack them together to have all possible spectral features, e.g., **HTCSA**, **Catch-22**, etc.
- Give state-of-the-art models the animal inputs anyway and hope it works. The issue with this is that these models are often pre-trained on audio/speech down-sampled to 8 kHz bandwidth (information beyond this frequency is eliminated), and animal vocalizations often start at 8-10 kHz.

The general problem is that it depends a lot on the animal and its spectral range. It's impossible to have a set of features for all animals because the diversity is too great. But if it's just bird songs, well, birds are the simplest animals to classify in bioacoustics in my opinion. You just need to visualize the spectrograms, and you can recognize the patterns yourself by eye. I don't know what is considered the best feature today, but I don't think it's very complex.
