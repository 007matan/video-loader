import os
from src.audio_loader import load_audio_file
from src.audio_analyzer import analyze_audio_for_notes_pyin, analyze_audio_for_chords_cqt


def run_pantico_analyzer():
    """
    Prompts the user for an audio file path, loads it, analyzes it for Pantico notes,
    and prints the detected musical events.
    """
    while True:
        file_path = input(
            "Please enter the full path to your audio file (WAV or MP3), or 'exit' to quit: ")

        if file_path.lower() == 'exit':
            print("Exiting program.")
            break

        file_path = file_path.strip('"\'')

        if not os.path.isabs(file_path):
            print("Please provide a full (absolute) path to the file.")
            continue

        audio_data, sample_rate, data_type = load_audio_file(file_path)

        if audio_data is not None:
            print("\nAudio loaded successfully!")
            print(f"NumPy array shape: {audio_data.shape}")
            print(f"Sample Rate: {sample_rate} Hz")

            # --- Choose your analysis method ---
            # Option 1: Use PYIN for single note detection (as before)
            # music_events = analyze_audio_for_notes_pyin(audio_data, sample_rate, tolerance_hz=7)
            #
            # print(f"\n--- Detected Musical Events (PYIN, {len(music_events)}): ---")
            # if music_events:
            #     for event in music_events[:10]:
            #         print(f"Note: {event['note']}, Start: {event['start_time']:.2f}s, End: {event['end_time']:.2f}s")
            #     if len(music_events) > 10:
            #         print("...")
            # else:
            #     print("No musical events detected (PYIN) within the specified pitch range or tolerance.")

            # Option 2: Use CQT for chord detection (new functionality)
            C_db, cqt_frequencies, cqt_times, detected_pitches_per_frame = \
                analyze_audio_for_chords_cqt(audio_data, sample_rate)

            if C_db is not None:
                print("\n--- CQT Analysis Results (Partial): ---")
                print(
                    f"First 5 dominant frequencies in first frame: {detected_pitches_per_frame[0]['dominant_frequencies']}")
                print(f"Time of first frame: {detected_pitches_per_frame[0]['time']:.2f}s")
                print("\nNOTE: This is not full chord detection yet, just showing prominent frequencies per frame.")
                print("The CQT spectrogram (C_db) is ready for further processing.")

                # You can now pass C_db, cqt_frequencies, cqt_times to a visualization module
                # or build out the full chord detection logic here or in audio_analyzer.py

            else:
                print("CQT analysis failed.")
            # --- End analysis method choice ---

        else:
            print("Failed to load audio file. Please try again.")


if __name__ == "__main__":
    run_pantico_analyzer()