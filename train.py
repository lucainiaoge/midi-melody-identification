import os
import argparse
import numpy as np
import pretty_midi
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from midi_features import get_track_features

def parse_melody_channel_txt(txt_path):
    f = open(txt_path, "r")
    lines = [line.rstrip() for line in f]
    filenames = []
    channels = []
    for line in lines:
        this_filename, this_channel = line.split(",")
        filenames.append(this_filename)
        channels.append(int(this_channel))
    f.close()
    return filenames, channels

def get_dataset(label_txt_path, midi_dir):
    all_track_features = None
    all_labels = None
    filenames, channels = parse_melody_channel_txt(label_txt_path)
    for i_file in range(len(filenames)):
        midi_path = os.path.join(midi_dir, filenames[i_file])
        if not os.path.exists(midi_path):
            continue
        pretty_midi_features = pretty_midi.PrettyMIDI(midi_path)
        num_channels = len(pretty_midi_features.instruments)
        labels = np.zeros(num_channels, bool)
        labels[channels[i_file]] = True
        track_features = get_track_features(pretty_midi_features)
        all_labels = labels if all_labels is None else np.concatenate((all_labels, labels), axis=0)
        all_track_features = track_features if all_track_features is None else np.concatenate((all_track_features, track_features), axis=0)
    return all_track_features, all_labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args in midi melody track classifier training.')
    parser.add_argument(
        '-label-txt-path', type=str, default="melody_channel_ids_new.txt",
        help='A path to the melody track labels'
    )
    parser.add_argument(
        '-midi-dir', type=str, default="dataset_cmu_melody/train",
        help='The directory where training midi files locate'
    )
    parser.add_argument(
        '-out-dir', type=str, default=".",
        help='The directory where trained model is saved'
    )
    args = parser.parse_args()

    print("Loading data features...")
    X, y = get_dataset(args.label_txt_path, args.midi_dir)

    print("Training model...")
    pipeline = Pipeline([
        ('normalizer', StandardScaler()), 
        ('gb_classifier', HistGradientBoostingClassifier(max_iter=100))
    ]).fit(X, y)
    scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=5)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), 'HistGradientBoosting'))

    save_path = os.path.join(args.out_dir, "classifier_model.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Saved model to {save_path}")
    