import os
import argparse
import numpy as np
import pretty_midi
import pickle

from midi_features import get_track_features

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args in midi melody track classification')
    parser.add_argument(
        '-midi-path', type=str,
        help='The path to the input midi file'
    )
    parser.add_argument(
        '-model-path', type=str, default="classifier_model.pkl",
        help='The path to classifier sklearn model pipeline (pickle format)'
    )
    args = parser.parse_args()
    
    with open(args.model_path, 'rb') as f:
        pipeline = pickle.load(f)

    pretty_midi_features = pretty_midi.PrettyMIDI(args.midi_path)
    track_features = get_track_features(pretty_midi_features)
    
    y_pred = pipeline.predict_proba(track_features)[:,1]
    channel_pred = np.argmax(y_pred)
    print("Predicted melody channel:", channel_pred)