import librosa
import numpy as np


def analyze_audio_for_notes(audio_data, sample_rate, tolerance_hz=7):
    """
    Analyzes audio data to detect musical notes using pitch detection (PYIN)
    and maps them to predefined Pantico notes.

    :param audio_data: NumPy array of audio samples (mono, float32, normalized to [-1, 1]).
    :param sample_rate: The sample rate of the audio.
    :param tolerance_hz: The maximum frequency difference (in Hz) to consider a match for a Pantico note.
    :return: A list of dictionaries, where each dictionary represents a detected musical event
             with 'note', 'start_time', and 'end_time'.
    """
    if audio_data is None:
        print("Cannot perform audio analysis: no audio data provided.")
        return []

    print("\n--- Starting Audio Analysis (Pitch Detection) ---")

    # Define the frequency range of your Pantico (B2 to A4)
    # These are standard MIDI note to Hz conversions
    fmin_hz = librosa.note_to_hz('B2')
    fmax_hz = librosa.note_to_hz('A4')

    # Perform fundamental frequency estimation (PYIN)
    # The output variables are f0 (pitch estimates), voiced_flag (boolean array indicating voiced frames),
    # and voiced_probs (probability of a frame being voiced).
    f0, voiced_flag, voiced_probs = librosa.pyin(  # Corrected: using voiced_flag here
        audio_data,
        sr=sample_rate,
        fmin=fmin_hz,
        fmax=fmax_hz,
        frame_length=2048,  # Analysis window size
        hop_length=512,  # Hop size between windows (affects time resolution)
    )

    # Create a time axis corresponding to the f0 estimates
    times = librosa.times_like(f0, sr=sample_rate, hop_length=512)

    print(f"Detected {len(f0)} pitch estimates over {len(times)} time points.")

    # Your Pantico's exact notes and their frequencies
    # It's good to define these once and reuse.
    phan_notes_freqs = {
        'B2': librosa.note_to_hz('B2'),
        'A3': librosa.note_to_hz('A3'),
        'B3': librosa.note_to_hz('B3'),
        'C#4': librosa.note_to_hz('C#4'),
        'D4': librosa.note_to_hz('D4'),
        'E4': librosa.note_to_hz('E4'),
        'F#4': librosa.note_to_hz('F#4'),
        'A4': librosa.note_to_hz('A4')
    }

    detected_music_events = []
    last_detected_note = None
    note_start_time = None

    for i, freq_estimate in enumerate(f0):
        current_time = times[i]

        # Check if the frame is considered voiced (not silence or noise)
        if voiced_flag[i]:
            best_match_note = None
            min_diff = float('inf')

            # Find the closest Pantico note within tolerance
            for note_name, target_freq in phan_notes_freqs.items():
                diff = abs(freq_estimate - target_freq)
                if diff < min_diff and diff < tolerance_hz:
                    min_diff = diff
                    best_match_note = note_name

            if best_match_note:
                # If the detected note is different from the last one, or this is the first note
                if best_match_note != last_detected_note:
                    # If there was a previous note, add it to our events list
                    if last_detected_note is not None:
                        detected_music_events.append({
                            'note': last_detected_note,
                            'start_time': note_start_time,
                            'end_time': current_time
                        })
                    # Start a new note event
                    last_detected_note = best_match_note
                    note_start_time = current_time
        else:  # If the frame is unvoiced (silence/noise), end the current note event
            if last_detected_note is not None:
                detected_music_events.append({
                    'note': last_detected_note,
                    'start_time': note_start_time,
                    'end_time': current_time
                })
                last_detected_note = None
                note_start_time = None

    # After the loop, add any remaining active note that extends to the end of the audio
    if last_detected_note is not None:
        detected_music_events.append({
            'note': last_detected_note,
            'start_time': note_start_time,
            'end_time': times[-1]
        })

    return detected_music_events