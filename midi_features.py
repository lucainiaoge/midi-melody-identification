import numpy as np
import pretty_midi

MIDI_INSTRUMENTS = {
    "piano": np.arange(1, 9) - 1,
    "chromatic_percussion": np.arange(9, 17) - 1,
    "organ": np.arange(17, 25) - 1,
    "guitar": np.arange(25, 33) - 1,
    "bass": np.arange(33, 41) - 1,
    "strings": np.arange(41, 49) - 1,
    "ensemble": np.arange(49, 57) - 1,
    "brass": np.arange(57, 65) - 1,
    "reed": np.arange(65, 73) - 1,
    "pipe": np.arange(73, 81) - 1,
    "synth": np.arange(81, 105) - 1,
    "ethnic": np.arange(105, 112) - 1,
    "others": np.arange(112, 128) - 1,
}

def instrument_midi_id_to_type_string(midi_id):
    assert 0 <= midi_id <= 127
    for instrument_type in MIDI_INSTRUMENTS:
        if midi_id in MIDI_INSTRUMENTS[instrument_type]:
            return instrument_type

def extract_pretty_midi_features(midi_filepath):
    return pretty_midi.PrettyMIDI(midi_filepath)

def extract_pretty_midi_features_multiple(midi_filepaths):
    return [extract_pretty_midi_features(midi_filepath) for midi_filepath in midi_filepaths]

def get_num_notes(pretty_midi_features):
    get_onsets = pretty_midi_features.get_onsets()
    return len(get_onsets)

def get_note_densities(pretty_midi_features):
    total_duration = pretty_midi_features.get_end_time()
    note_densities = []
    # duration_densities = []
    for instrument in pretty_midi_features.instruments:
        note_density_this_instrument = 0
        # duration_density_this_instrument = 0
        for note in instrument.notes:
            note_density_this_instrument += 1
            # duration_density_this_instrument += note.end - note.start
        note_density_this_instrument /= total_duration
        # duration_density_this_instrument /= total_duration
        note_densities.append(note_density_this_instrument)
        # duration_densities.append(duration_density_this_instrument)
    return np.array(note_densities)#, np.array(duration_densities)

def get_pitch_velocity_tracks(pretty_midi_features):
    pitches = []
    velocities = []
    for instrument in pretty_midi_features.instruments:
        pitches_this_instrument = []
        velocities_this_instrument = []
        for note in instrument.notes:
            pitches_this_instrument.append(note.pitch)
            velocities_this_instrument.append(note.velocity)
        pitches.append(np.array(pitches_this_instrument))
        velocities.append(np.array(velocities_this_instrument))
    return pitches, velocities # pitches.mean(), pitches.std(), velocities.mean(), velocities.std()

def get_polyphony_rates_and_duration_densities(pretty_midi_features, fs=20):
    polyphony_rates = []
    activated_secs = []
    duration_densities = []
    total_duration = pretty_midi_features.get_end_time()
    for i_channel, instrument in enumerate(pretty_midi_features.instruments):
        instrument_roll = (pretty_midi_features.instruments[i_channel].get_piano_roll(fs=fs) > 0).astype(int)
        activations = instrument_roll.sum(0)
        non_zero_time = activations[activations > 0]
        activated_sec = len(non_zero_time) / fs
        if len(non_zero_time) > 0:
            polyphony_rates.append(non_zero_time.sum() / len(non_zero_time))
        else:
            polyphony_rates.append(0)
        activated_secs.append(activated_sec)
        duration_densities.append(activated_sec / total_duration)
    return np.array(polyphony_rates), np.array(activated_secs), np.array(duration_densities)
    
# inter-onset interval
def get_ioi_tracks(pretty_midi_features, eps = 0.075):
    iois = []
    for instrument in pretty_midi_features.instruments:
        iois_this_instrument = []
        total_notes = len(instrument.notes)
        if total_notes <= 1:
            iois.append(np.array([0]))
            continue
        i = 0
        while i < len(instrument.notes):
            j = i + 1
            while j < total_notes and (instrument.notes[j].start - instrument.notes[j-1].start) < eps:
                j = j + 1
            if j < total_notes:
                iois_this_instrument.append(instrument.notes[j].start - instrument.notes[i].start)
            i = j
        iois.append(np.array(iois_this_instrument))
    return iois

def get_track_features(pretty_midi_features):
    polyphony_rates, activated_secs, duration_densities = get_polyphony_rates_and_duration_densities(pretty_midi_features)
    note_densities = get_note_densities(pretty_midi_features)
    pitches, velocities = get_pitch_velocity_tracks(pretty_midi_features)
    iois = get_ioi_tracks(pretty_midi_features)
    num_tracks = len(pretty_midi_features.instruments)
    features = []
    for i_track in range(num_tracks):
        this_polyphony_rate = polyphony_rates[i_track]
        this_note_density, this_duration_density = note_densities[i_track], duration_densities[i_track]
        this_pitch_mean, this_pitch_std = pitches[i_track].mean(), pitches[i_track].std()
        this_velocity_mean, this_velocity_std = velocities[i_track].mean(), velocities[i_track].std()
        this_ioi_mean, this_ioi_std = iois[i_track].mean(), iois[i_track].std()

        select_mask = np.ones(num_tracks, bool)
        select_mask[i_track] = False
        pitches_others = np.concatenate(pitches[:i_track] + pitches[i_track+1:])
        velocities_others = np.concatenate(velocities[:i_track] + velocities[i_track+1:])
        iois_others = np.concatenate(iois[:i_track] + iois[i_track+1:])

        others_polyphony_rate = (note_densities[select_mask] * activated_secs[select_mask]).sum() / activated_secs[select_mask].sum()
        others_note_density, others_duration_density = note_densities[select_mask].mean(), duration_densities[select_mask].mean()
        others_pitch_mean, others_pitch_std = pitches_others.mean(), pitches_others.std()
        others_velocity_mean, others_velocity_std = velocities_others.mean(), velocities_others.std()
        others_ioi_mean, others_ioi_std = iois_others.mean(), iois_others.std()

        features.append(
            [
                this_polyphony_rate,
                this_note_density, this_duration_density, 
                this_pitch_mean, this_pitch_std, 
                this_velocity_mean, this_velocity_std, 
                this_ioi_mean, this_ioi_std, 

                others_polyphony_rate,
                others_note_density, others_duration_density,
                others_pitch_mean, others_pitch_std, 
                others_velocity_mean, others_velocity_std, 
                others_ioi_mean, others_ioi_std,
            ]
        )
    return np.array(features)