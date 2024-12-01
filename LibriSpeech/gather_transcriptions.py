import os

def gather_librispeech_transcriptions(root_folder):
    for dataset_folder in os.listdir(root_folder):
        dataset_path = os.path.join(root_folder, dataset_folder)
        
        if not os.path.isdir(dataset_path) or dataset_folder == 'data':
            continue
        
        output_file = os.path.join(root_folder, "data", f"{dataset_folder}.txt")
        
        with open(output_file, "w", encoding="utf-8") as outfile:
            for subdir, _, files in os.walk(dataset_path):
                for file in files:
                    if file.endswith(".txt"):
                        txt_file_path = os.path.join(subdir, file)
                        
                        with open(txt_file_path, "r", encoding="utf-8") as infile:
                            for line in infile:
                                parts = line.strip().split(" ", 1)
                                if len(parts) != 2:
                                    continue
                                
                                audio_id, transcription = parts
                                formatted_transcription = transcription.lower().replace(" ", "_")
                                
                                outfile.write(f"{audio_id} {formatted_transcription}\n")
        
        print(f"Processed dataset: {dataset_folder}, output saved to: {output_file}")

if __name__ == "__main__":
    gather_librispeech_transcriptions(os.getcwd())
