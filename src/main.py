import os
# Import functions from our new modules
from src.audio_loader import load_audio_file
from src.audio_analyzer import analyze_audio_for_notes


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

            # Pass the tolerance to the analysis function
            music_events = analyze_audio_for_notes(audio_data, sample_rate,
                                                   tolerance_hz=7)  # Using your custom tolerance

            print(f"\n--- Detected Musical Events ({len(music_events)}): ---")
            if music_events:
                for event in music_events[:10]:
                    print(f"Note: {event['note']}, Start: {event['start_time']:.2f}s, End: {event['end_time']:.2f}s")
                if len(music_events) > 10:
                    print("...")
            else:
                print("No musical events detected within the specified pitch range or tolerance.")

        else:
            print("Failed to load audio file. Please try again.")


if __name__ == "__main__":
    run_pantico_analyzer()