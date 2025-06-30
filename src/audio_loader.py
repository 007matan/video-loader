import os
from pydub import AudioSegment
import soundfile as sf
import numpy as np


def load_audio_file(file_path):
    """
    Loads an audio file (WAV or MP3) and returns its data as a NumPy array, sample rate, and data type string.
    The audio data is converted to mono and normalized for Librosa compatibility.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at path: {file_path}")
        return None, None, None

    file_extension = os.path.splitext(file_path)[1].lower()

    try:
        if file_extension == '.wav':
            data, samplerate = sf.read(file_path)

            # Ensure data is float32 for Librosa
            if data.dtype != np.float32:
                data = data.astype(np.float32)

            # Convert to mono if stereo (Librosa generally expects mono)
            if data.ndim > 1 and data.shape[1] == 2:  # If stereo (shape is (samples, 2))
                data = data.mean(axis=1)  # Simple conversion to mono by averaging channels

            print(f"\n--- WAV File Details: {os.path.basename(file_path)} ---")
            print(f"Sample Rate: {samplerate} Hz")
            # Correctly display number of channels after potential mono conversion
            print(f"Number of Channels: {1 if data.ndim == 1 else data.shape[1]}")
            print(f"Duration (approx): {len(data) / samplerate:.2f} seconds")
            return data, samplerate, 'numpy'

        elif file_extension == '.mp3':
            audio = AudioSegment.from_mp3(file_path)
            print(f"\n--- MP3 File Details: {os.path.basename(file_path)} ---")
            print(f"Sample Rate: {audio.frame_rate} Hz")
            print(f"Bit Rate: {audio.frame_rate * audio.sample_width * audio.channels / 1000} kbps (approx)")
            print(f"Number of Channels: {audio.channels}")
            print(f"Duration: {audio.duration_seconds:.2f} seconds")

            data = np.array(audio.get_array_of_samples(), dtype=np.float32)

            # If stereo, convert to mono
            if audio.channels == 2:
                data = data.reshape((-1, 2))  # Reshape to (samples, 2 channels)
                data = data.mean(axis=1)  # Convert to mono by averaging the two channels

            # Normalize data to [-1.0, 1.0] for Librosa
            data /= (2 ** 15)

            return data, audio.frame_rate, 'numpy'

        else:
            print(f"Unsupported file format: {file_extension}. Only WAV and MP3 are supported.")
            return None, None, None
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        print("Ensure FFmpeg is installed and accessible in your system's PATH for MP3 files.")
        return None, None, None