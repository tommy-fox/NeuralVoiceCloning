import os
import argparse
import random
from tqdm import tqdm
from phonemizer import phonemize

def format_vctk(vctk_root, output_dir, val_split=0.05):
    txt_dir = os.path.join(vctk_root, 'txt')
    audio_dir = os.path.join(vctk_root, 'wav24')
    speakers = sorted([d for d in os.listdir(txt_dir) if os.path.isdir(os.path.join(txt_dir, d))])

    all_entries = []

    print("Collecting all text entries...")
    for speaker in tqdm(speakers, desc="Collecting text"):
        speaker_id = speaker
        speaker_txt_dir = os.path.join(txt_dir, speaker)
        speaker_audio_dir = os.path.join(audio_dir, speaker)

        txt_files = sorted([f for f in os.listdir(speaker_txt_dir) if f.endswith('.txt')])
        for txt_file in txt_files:
            base_name = os.path.splitext(txt_file)[0]
            wav_file_mic1 = os.path.join(speaker_audio_dir, base_name + '_mic1.flac')
            wav_file_mic2 = os.path.join(speaker_audio_dir, base_name + '_mic2.flac')
            txt_path = os.path.join(speaker_txt_dir, txt_file)

            if not os.path.exists(wav_file_mic1) or not os.path.exists(wav_file_mic2):
                continue

            with open(txt_path, 'r', encoding='utf-8') as f:
                raw_text = f.read().strip()

            all_entries.append({
                "speaker": speaker_id,
                "text": raw_text,
                "mic1": os.path.relpath(wav_file_mic1, vctk_root),
                "mic2": os.path.relpath(wav_file_mic2, vctk_root)
            })

    print(f"Phonemizing {len(all_entries)} entries...")
    raw_texts = [entry['text'] for entry in all_entries]
    phonemes = phonemize(
        raw_texts,
        language='en-us',
        backend='espeak',
        strip=True,
        preserve_punctuation=True,
        with_stress=False,
    )

    print("Rebuilding train/val/OOD entries with phonemized text...")
    train_entries = []
    val_entries = []
    ood_entries = []

    for entry, phoneme_text in zip(all_entries, phonemes):
        line_mic1 = f"{entry['mic1']}|{phoneme_text}|{entry['speaker']}"
        line_mic2 = f"{entry['mic2']}|{phoneme_text}|{entry['speaker']}"
        ood_entries.extend([line_mic1, line_mic2])

        if random.random() < val_split:
            val_entries.extend([line_mic1, line_mic2])
        else:
            train_entries.extend([line_mic1, line_mic2])

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_entries))

    with open(os.path.join(output_dir, 'val.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_entries))

    with open(os.path.join(output_dir, 'OOD_texts.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(ood_entries))

    print(f"✅ Done. Saved {len(train_entries)} train, {len(val_entries)} val, and {len(ood_entries)} OOD entries.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert VCTK to StyleTTS2 format with phonemization")
    parser.add_argument('--vctk_root', type=str, required=True, help='Root directory of VCTK')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for metadata files')
    parser.add_argument('--val_split', type=float, default=0.05, help='Validation split ratio')
    args = parser.parse_args()

    format_vctk(args.vctk_root, args.output_dir, args.val_split)
