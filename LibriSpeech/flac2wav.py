import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from pydub import AudioSegment

def convert_flac_to_wav(file_path):
    """
    Convert a .flac file to .wav and delete the original .flac file.

    Args:
        file_path (str): Full path of the .flac file.
    """
    try:
        wav_path = file_path.with_suffix('.wav')
        audio = AudioSegment.from_file(file_path, format="flac")
        audio.export(wav_path, format="wav")
        #print(f"Converted: {file_path} -> {wav_path}")
        
        # Delete the original .flac file
        file_path.unlink()
        #print(f"Deleted original file: {file_path}")
    except Exception as e:
        print(f"Error converting {file_path}: {e}")

def process_folder(root_folder):
    """
    Finds all .flac files in the directory and converts to .wav using multiple processes.

    Args:
        root_folder (str): Root directory path.
    """
    root_folder = Path(root_folder)

    flac_files = list(root_folder.rglob("*.flac"))
    if not flac_files:
        print("No .flac files found.")
        return

    print(f"Number of .flac files found: {len(flac_files)}")

    with Pool(cpu_count()) as pool:
        pool.map(convert_flac_to_wav, flac_files)

if __name__ == "__main__":
    process_folder(os.getcwd())