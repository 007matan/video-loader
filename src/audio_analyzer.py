import librosa
import numpy as np

# A good place to define your Pantico notes and frequencies once
# These are standard MIDI note to Hz conversions
PANTICO_NOTES_FREQS = {
    'B2': librosa.note_to_hz('B2'),
    'A3': librosa.note_to_hz('A3'),
    'B3': librosa.note_to_hz('B3'),
    'C#4': librosa.note_to_hz('C#4'),
    'D4': librosa.note_to_hz('D4'),
    'E4': librosa.note_to_hz('E4'),
    'F#4': librosa.note_to_hz('F#4'),
    'A4': librosa.note_to_hz('A4')
}


def analyze_audio_for_notes_pyin(audio_data, sample_rate, tolerance_hz=7):
    """
    Analyzes audio data to detect musical notes using pitch detection (PYIN)
    and maps them to predefined Pantico notes. This is good for single-note melodies.

    :param audio_data: NumPy array of audio samples (mono, float32, normalized to [-1, 1]).
    :param sample_rate: The sample rate of the audio.
    :param tolerance_hz: The maximum frequency difference (in Hz) to consider a match for a Pantico note.
    :return: A list of dictionaries, where each dictionary represents a detected musical event
             with 'note', 'start_time', and 'end_time'.
    """
    if audio_data is None:
        print("Cannot perform audio analysis: no audio data provided.")
        return []

    print("\n--- Starting Audio Analysis (Pitch Detection using PYIN) ---")

    fmin_hz = PANTICO_NOTES_FREQS['B2']
    fmax_hz = PANTICO_NOTES_FREQS['A4']

    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio_data,
        sr=sample_rate,
        fmin=fmin_hz,
        fmax=fmax_hz,
        frame_length=2048,
        hop_length=512,
    )

    times = librosa.times_like(f0, sr=sample_rate, hop_length=512)

    print(f"Detected {len(f0)} pitch estimates over {len(times)} time points.")

    detected_music_events = []
    last_detected_note = None
    note_start_time = None

    for i, freq_estimate in enumerate(f0):
        current_time = times[i]

        if voiced_flag[i]:
            best_match_note = None
            min_diff = float('inf')

            for note_name, target_freq in PANTICO_NOTES_FREQS.items():
                diff = abs(freq_estimate - target_freq)
                if diff < min_diff and diff < tolerance_hz:
                    min_diff = diff
                    best_match_note = note_name

            if best_match_note:
                if best_match_note != last_detected_note:
                    if last_detected_note is not None:
                        detected_music_events.append({
                            'note': last_detected_note,
                            'start_time': note_start_time,
                            'end_time': current_time
                        })
                    last_detected_note = best_match_note
                    note_start_time = current_time
        else:
            if last_detected_note is not None:
                detected_music_events.append({
                    'note': last_detected_note,
                    'start_time': note_start_time,
                    'end_time': current_time
                })
                last_detected_note = None
                note_start_time = None

    if last_detected_note is not None:
        detected_music_events.append({
            'note': last_detected_note,
            'start_time': note_start_time,
            'end_time': times[-1]
        })

    return detected_music_events


def analyze_audio_for_chords_cqt(audio_data, sample_rate, n_bins=84, bins_per_octave=12):
    """
    Analyzes audio data using Constant-Q Transform (CQT) to identify prominent frequencies
    which can then be used for chord detection.

    :param audio_data: NumPy array of audio samples (mono, float32, normalized to [-1, 1]).
    :param sample_rate: The sample rate of the audio.
    :param n_bins: Number of frequency bins (notes) in the CQT.
    :param bins_per_octave: Number of frequency bins per octave. 12 for standard Western music.
    :return: A tuple: (CQT_spectrogram, CQT_frequencies, CQT_times)
             - CQT_spectrogram: The magnitude spectrogram (NumPy array).
             - CQT_frequencies: Frequencies corresponding to each CQT bin.
             - CQT_times: Time points corresponding to each frame.
    """
    if audio_data is None:
        print("Cannot perform CQT analysis: no audio data provided.")
        return None, None, None

    print("\n--- Starting Audio Analysis (Chord Detection using CQT) ---")

    # Define the lowest frequency for CQT, typically C1 or A0 for musical analysis
    # Let's align with your Pantico's lowest note, B2, or slightly below it to catch its harmonics
    fmin_cqt = librosa.note_to_hz('A2')

    # Compute CQT spectrogram
    # C = librosa.cqt(y=audio_data, sr=sample_rate, hop_length=512, fmin=fmin_cqt,
    #                  n_bins=n_bins, bins_per_octave=bins_per_octave)
    # The magnitude spectrogram (absolute value of the complex CQT output)
    # Convert to decibels for better visualization and analysis (logarithmic scale)
    C_db = librosa.amplitude_to_db(np.abs(
        librosa.cqt(y=audio_data, sr=sample_rate, hop_length=512, fmin=fmin_cqt, n_bins=n_bins,
                    bins_per_octave=bins_per_octave)), ref=np.max)

    # Get frequencies and times corresponding to the CQT
    cqt_frequencies = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin_cqt, bins_per_octave=bins_per_octave)
    cqt_times = librosa.frames_to_time(np.arange(C_db.shape[1]), sr=sample_rate, hop_length=512)

    print(f"CQT Spectrogram shape: {C_db.shape} (Bins: {C_db.shape[0]}, Frames: {C_db.shape[1]})")
    print(f"CQT covers frequencies from {cqt_frequencies.min():.2f} Hz to {cqt_frequencies.max():.2f} Hz.")

    # At this point, C_db contains the spectrogram.
    # The next step would be to extract dominant pitches from C_db at each time frame.
    # This is more complex than simple F0 detection as we look for multiple strong frequencies.

    # --- Example of finding dominant pitches (NOT a full chord detection yet, but a start) ---
    detected_pitches_per_frame = []
    for t_idx in range(C_db.shape[1]):  # Iterate through each time frame
        frame_spectrum = C_db[:, t_idx]

        # Find peaks in the spectrum of the current frame
        # We look for bins that are significantly louder than their neighbors
        # A simple thresholding or peak picking algorithm can be applied here

        # For simplicity, let's find the top N loudest frequencies in this frame
        # This is a very basic approach and would need refinement for real chord detection
        top_n_peaks_indices = np.argsort(frame_spectrum)[::-1][:5]  # Get indices of top 5 loudest bins

        # Filter out very quiet peaks (e.g., below a certain dB threshold)
        threshold_db = -60  # Adjust this threshold based on your audio's dynamics

        dominant_freqs_in_frame = []
        for idx in top_n_peaks_indices:
            if frame_spectrum[idx] > threshold_db:
                dominant_freqs_in_frame.append(cqt_frequencies[idx])

        detected_pitches_per_frame.append({
            'time': cqt_times[t_idx],
            'dominant_frequencies': dominant_freqs_in_frame
        })
    # --- End example ---

    # For now, let's just return the spectrogram and related info for visualization/debug
    # In a later step, we'll build the chord detection logic based on dominant_frequencies
    return C_db, cqt_frequencies, cqt_times, detected_pitches_per_frame

