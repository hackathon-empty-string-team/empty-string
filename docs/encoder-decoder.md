## Current standings

- no models trained at this point
- for implementation details check the code


## rough overview

- audio file is read, resampled and mel spectrograms are computed
- there is overlap, e.g. one recording is put into 11 frames (here method differs from vggish where no overlap is computed, we can discuss this)
- IMO it is better to have overlap in order to not "cut" a call in half
- currently, it is implemented for SoundMeters dataset and uses the length of each recording, resulting in roughly 11 overlapping samples. This is hacky and needs to be adapted but i dont have the capacity right now. 
- audio dataset output is then fed in to the models (depending on if we change the sampling, we need to update model archtitecture as well)
- then the model should be trained on the data, on my laptop this is not possible. maybe we need to port this to a google colab or supercomputer setup. 
- after training, the saved model can be used with the `feature_extraction` script. the output currently, `(11, 128)` so essentially for each frame we get a vector with 128 dimensions. As long as we only use soundmeters, this is "okayish" but i guess there is room for improvement .
- the extracted feature vector could then be used for clustering
