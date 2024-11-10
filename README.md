# midi-melody-identification
A simple method to identify the melody channel/instrument in a multi-track midi file

## Usage Example

```
python inference.py -midi-path dataset_cmu_melody/test/california.mid
```

## Basic method
I improve the melody probability calculation method of [Melody Identification in Standard MIDI Files](https://www.cs.cmu.edu/~rbd/papers/melody-identification-midi-smc2019.pdf) in several ways:
- We add additional features including polyphony rate and note activation density
- We calculate the melody probability (binary classification) providing both the features of current track and the average features of other tracks in the same midi file
- After model selection, we adopt sklearn HistGradientBoostingClassifier as the base model

In case you are curious about the model selection: checkout "2-model_selection.ipynb"

In case you want to train the model:
```
python train.py -label-txt-path melody_channel_ids_new.txt -midi-dir dataset_cmu_melody/train
```
I have already done this, which saves "classifier_model.pkl".

## Dataset
I use a small amount of multi-track midi files to optimize our model parameter. The midi files are taken from the [CMU Computer Music Analysis Dataset](https://www.cs.cmu.edu/~music/data/melody-identification/).

However, I observe that the melody channel labels in this dataset is not well-organized (several ways are tried, including opening the midi files with Musescore and pretty_midi, but the melody track cannot match the labelled one for many midi files in this dataset). Therefore, I labelled the melody channels myself through listening (opening the midi with Musescore and mark the track id which contains melody, then double-check the melody track by printing the piano roll of the marked track).

In this repo, "melody_channel_ids_new.txt" is provided, a correctness-guarenteed melody channel label file for the CMU Computer Music Analysis Dataset training set (in the original dataset, the corresponding label file is named "melody_label.txt" in the train folder).

In this repo, a notebook "1-melody_track_identification.ipynb" is provided for comparison with my new labels and the original labels.
