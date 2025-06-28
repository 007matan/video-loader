import os
from pydub import AudioSegment
import soundfile as sf
import numpy as np
import librosa  # הוספתי ייבוא של librosa - חיוני לשלבים הבאים


def load_audio_file(file_path):
    """
    :param file_path: an audio file (WAV or MP3)
    :return: audio data (NumPy array for WAV, AudioSegment for MP3), sample rate, and data type string
    """
    if not os.path.exists(file_path):  # תיקון: os.path.exist -> os.path.exists
        print(f"Error: File not found at path: {file_path}")
        return None, None, None

    file_extension = os.path.splitext(file_path)[1].lower()

    try:
        if file_extension == '.wav':
            data, samplerate = sf.read(file_path)
            print(f"\n--- WAV File Details: {os.path.basename(file_path)} ---")
            print(f"Sample Rate: {samplerate} Hz")
            print(f"Number of Channels: {data.shape[1] if data.ndim > 1 else 1}")
            print(f"Duration (approx): {len(data) / samplerate:.2f} seconds")
            return data, samplerate, 'numpy'  # NumPy array

        elif file_extension == '.mp3':
            audio = AudioSegment.from_mp3(file_path)
            print(f"\n--- MP3 File Details: {os.path.basename(file_path)} ---")
            print(f"Sample Rate: {audio.frame_rate} Hz")
            print(f"Bit Rate: {audio.frame_rate * audio.sample_width * audio.channels / 1000} kbps (approx)")
            print(f"Number of Channels: {audio.channels}")
            print(f"Duration: {audio.duration_seconds:.2f} seconds")
            # In order to use Librosa in the future, I should convert the pydub.AudioSegment to a NumPy array
            # And to change the samplerate to fix Librosa
            # Librosa prefers floating point arrays
            data = np.array(audio.get_array_of_samples(), dtype=np.float32)
            # If the Audio is in Sterao, Librosa expect Mono Signal
            if audio.channels == 2:
                data = data.reshape((-1, 2)).T  # shape (2, samples)
                data = data[0] + data[1]  # convert to basic mono
                data /= 2.0  # normalize


            # normalize fo Librosa [-1.0, 1.0]
            data /= (2 ** 15)  # For 16-bit audio samples (get_array_of_samples returns most of the times int 16)

            return data, audio.frame_rate, 'numpy'  # return NumPy array also for MP3


        else:
            print(f"Unsupported file format: {file_extension}. Only WAV and MP3 are supported.")
            return None, None, None
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        print("Ensure FFmpeg is installed and accessible in your system's PATH for MP3 files.")
        return None, None, None


def run_audio_loader():
    """
    Prompts the user for an audio file path and loads it using the audio_processing module.
    """
    while True:
        file_path = input(
            "Please enter the full path to your audio file (WAV or MP3), or 'exit' to quit: ")

        if file_path.lower() == 'exit':
            print("Exiting program.")
            break

        # Strip potential quotes if the user drags and drops files on some terminals
        file_path = file_path.strip('"\'')

        if not os.path.isabs(file_path):
            print("Please provide a full (absolute) path to the file.")
            continue  # Ask again

        audio_data, sample_rate, data_type = load_audio_file(file_path)

        if audio_data is not None:
            print("\nAudio loaded successfully!")
            print(f"NumPy array shape: {audio_data.shape}")
            print(f"Sample Rate: {sample_rate} Hz")

        else:
            print("Failed to load audio file. Please try again.")





if __name__ == "__main__":
    run_audio_loader()