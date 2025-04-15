'''
resample.py resamples a folder of audio files to a target sample rate
original data directory, resampled data directory, and target sample rate
are provided by a config.yaml file
'''
import os, yaml, argparse
import torchaudio
from torchaudio.functional import resample

def resample_data(
        input_dir,
        output_dir,
        target_sample_rate,
        audio_file_extension
):
    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        relative_path = os.path.relpath(root, input_dir)
        output_subdir = os.path.join(output_dir, relative_path)
        os.makedirs(output_subdir, exist_ok=True)

        for filename in files:
            if not filename.lower().endswith(audio_file_extension):
                continue
            
            input_path = os.path.join(root, filename)
            output_path = os.path.join(output_subdir, filename)

            # Skip file if it's already been resampled
            if os.path.exists(output_path):
                continue

            audio_data, original_sample_rate = torchaudio.load(input_path)

            if original_sample_rate != target_sample_rate:
                audio_data = resample(audio_data, orig_freq=original_sample_rate, new_freq=target_sample_rate)

            torchaudio.save(output_path, audio_data, sample_rate=target_sample_rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resample Audio Dataset")
    parser.add_argument("--config_file", type=str, default="configs/config.yaml", help="Path to YAML config")
    args = parser.parse_args()
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    input_dir = config["dataset_path"]
    output_dir = config["preprocessed_dataset_path"]
    target_sample_rate = config["sample_rate"]
    audio_file_extension = config["data_file_extension"]

    resample_data(input_dir=input_dir, 
                  output_dir=output_dir, 
                  target_sample_rate=target_sample_rate, 
                  audio_file_extension=audio_file_extension)