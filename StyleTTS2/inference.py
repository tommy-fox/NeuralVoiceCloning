"""
inference.py — Voice cloning inference for fine-tuned StyleTTS2 with custom
speaker embeddings.

Given a directory of .wav files from the target speaker, this script:
  1. Runs every wav through the custom speaker encoder.
  2. Averages the per-utterance embeddings into a single speaker embedding.
  3. Uses that embedding (plus a prosodic reference pulled from one of the
     clips) to condition the fine-tuned StyleTTS2 pipeline.
  4. Synthesises the requested text and writes a .wav file.

Usage
-----
    python inference.py \\
        --text "Hello, this is a cloned voice." \\
        --ref_audio_dir /path/to/target_speaker_wavs/ \\
        --text_output cloned_output.wav

By default the script expects these files (create them via symlink or copy):
    inference_models/finetuned_styletts2.pth   # fine-tuned StyleTTS2 ckpt
    inference_models/speaker_encoder.pt        # custom speaker encoder ckpt

You can override any path with a CLI flag (see --help).
"""

import argparse
import glob
import os
import os.path as osp
import sys

import nltk
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torchaudio.functional import resample as ta_resample
import yaml
from nltk.tokenize import word_tokenize

# StyleTTS2 imports — must be run from the StyleTTS2 directory
from models import *  # noqa: F401,F403
from models_util import load_ASR_models, load_F0_models, build_model
from utils import *  # noqa: F401,F403
from text_utils import TextCleaner
from models.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

# Speaker encoder lives in a sibling directory — import it directly via
# importlib to avoid collision with StyleTTS2's own `models/` package.
import importlib.util
SPEAKER_ENCODING_DIR = osp.abspath(
    osp.join(osp.dirname(__file__), '..', 'SpeakerEncoding'))
_spec = importlib.util.spec_from_file_location(
    'speaker_encoder',
    osp.join(SPEAKER_ENCODING_DIR, 'models', 'speaker_encoder.py'))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
SpeakerEncoder = _mod.SpeakerEncoder


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_STYLETTS_CKPT = osp.join(
    osp.dirname(__file__), 'inference_models', 'finetuned_styletts2.pth')
DEFAULT_SPKENC_CKPT = osp.join(
    osp.dirname(__file__), 'inference_models', 'speaker_encoder.pt')
DEFAULT_CONFIG = osp.join(
    osp.dirname(__file__), 'configs', 'config_ft.yml')
DEFAULT_SPKENC_CONFIG = osp.join(
    SPEAKER_ENCODING_DIR, 'configs', 'config.yaml')


# ---------------------------------------------------------------------------
# Audio / mel helpers
# ---------------------------------------------------------------------------

STYLETTS_MEL = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
STYLETTS_MEAN, STYLETTS_STD = -4, 4

# Speaker encoder mel params (matches SpeakerEncoding/configs/config.yaml)
SPKENC_SR = 16000
SPKENC_N_FFT = 2048
SPKENC_WIN = 1600
SPKENC_HOP = 400
SPKENC_N_MELS = 80
SPKENC_N_FRAMES = 201  # fixed temporal length expected by the encoder

SPKENC_MEL = torchaudio.transforms.MelSpectrogram(
    sample_rate=SPKENC_SR,
    n_fft=SPKENC_N_FFT,
    win_length=SPKENC_WIN,
    hop_length=SPKENC_HOP,
    n_mels=SPKENC_N_MELS,
)


def load_audio(path, target_sr):
    """Load audio file, convert to mono, resample to target_sr. Returns numpy array."""
    wave, sr = torchaudio.load(path)
    # Convert to mono
    if wave.shape[0] > 1:
        wave = wave.mean(dim=0, keepdim=True)
    # Resample
    if sr != target_sr:
        wave = ta_resample(wave, orig_freq=sr, new_freq=target_sr)
    return wave.squeeze(0).numpy()


def trim_silence(wave, top_db=30):
    """Trim leading/trailing silence from a numpy audio array (replaces librosa.effects.trim)."""
    ref = np.max(np.abs(wave))
    if ref == 0:
        return wave
    threshold = ref * 10 ** (-top_db / 20)
    above = np.abs(wave) > threshold
    indices = np.where(above)[0]
    if len(indices) == 0:
        return wave
    return wave[indices[0]:indices[-1] + 1]


def audio_to_styletts_mel(wave):
    """Convert 24kHz numpy audio to StyleTTS2 normalised mel tensor [80, T]."""
    wave_tensor = torch.from_numpy(wave).float()
    mel = STYLETTS_MEL(wave_tensor)
    mel = (torch.log(1e-5 + mel) - STYLETTS_MEAN) / STYLETTS_STD
    return mel  # [80, T]


SPKENC_DURATION = 5.0  # seconds — must match SpeakerEncoding config sample_duration

def audio_to_spkenc_mel(wave):
    """
    Convert audio numpy array (already at SPKENC_SR) to a fixed-length mel
    tensor of shape [1, SPKENC_N_MELS, SPKENC_N_FRAMES] suitable for the
    speaker encoder.

    Matches the preprocessing in SpeakerEncoding/utils/preprocess_audio.py:
      1. Pad/truncate waveform to exactly SPKENC_DURATION seconds.
      2. Compute mel spectrogram.
      3. Z-score normalize (NOT log).
    """
    wave_tensor = torch.from_numpy(wave).float().unsqueeze(0)  # [1, samples]

    # Pad or truncate waveform to fixed duration (same as training preprocessing)
    max_samples = int(SPKENC_DURATION * SPKENC_SR)
    if wave_tensor.shape[1] > max_samples:
        wave_tensor = wave_tensor[:, :max_samples]
    elif wave_tensor.shape[1] < max_samples:
        padding = max_samples - wave_tensor.shape[1]
        wave_tensor = torch.nn.functional.pad(wave_tensor, (0, padding))

    # Compute mel and z-score normalize (matching preprocess_audio.py exactly)
    mel = SPKENC_MEL(wave_tensor).squeeze(0)  # [n_mels, T]
    mel = (mel - mel.mean()) / (mel.std() + 1e-8)

    return mel.unsqueeze(0)  # [1, n_mels, n_frames]


# ---------------------------------------------------------------------------
# Style embedding computation
# ---------------------------------------------------------------------------

def gather_wavs(ref_audio_dir):
    """Return a sorted list of every .wav file under ref_audio_dir (recursive)."""
    if not osp.isdir(ref_audio_dir):
        raise ValueError(f'ref_audio_dir is not a directory: {ref_audio_dir}')

    patterns = ('*.wav', '*.WAV')
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(osp.join(ref_audio_dir, '**', pat), recursive=True))
    paths = sorted(set(paths))

    if not paths:
        raise ValueError(
            f'No .wav files found in {ref_audio_dir} (searched recursively).')
    return paths


def compute_speaker_embedding(ref_audio_paths, speaker_encoder, device,
                              batch_clips=16):
    """
    Run every reference audio clip through the speaker encoder, then average
    the per-clip embeddings into a single [1, 128] speaker embedding.

    The clips are processed in small batches (batch_clips at a time) so that
    very large target-speaker directories don't blow up GPU memory.
    """
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(ref_audio_paths), batch_clips):
            chunk_paths = ref_audio_paths[i:i + batch_clips]
            mels = []
            for path in chunk_paths:
                wave = load_audio(path, SPKENC_SR)
                mel = audio_to_spkenc_mel(wave)  # [1, 80, 201]
                mels.append(mel)

            # Shape into [1, n_clips, 80, 201] — the encoder uses dim 1 as the
            # "reference samples" axis that the attention layer operates over.
            stacked = torch.stack(mels, dim=0)           # [n_clips, 1, 80, 201]
            stacked = stacked.squeeze(1).unsqueeze(0)    # [1, n_clips, 80, 201]
            stacked = stacked.to(device)

            emb = speaker_encoder(stacked)   # [1, n_clips, 128]
            all_embeddings.append(emb.squeeze(0).cpu())  # [n_clips, 128]

    all_embeddings = torch.cat(all_embeddings, dim=0)    # [N, 128]
    embedding = all_embeddings.mean(dim=0, keepdim=True) # [1, 128]
    return embedding.to(device)


def compute_prosodic_style(ref_audio_path, model, device):
    """
    Extract prosodic style from one reference clip using StyleTTS2's
    predictor_encoder. Returns ref_p of shape [1, 128].
    """
    wave = load_audio(ref_audio_path, 24000)
    audio = trim_silence(wave, top_db=30)
    mel = audio_to_styletts_mel(audio).to(device)  # [80, T]

    with torch.no_grad():
        ref_p = model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1))  # [1, 128]
    return ref_p


def pick_prosody_reference(ref_audio_paths, min_seconds=2.0, max_seconds=10.0):
    """
    Pick a reasonable clip to use as the prosodic reference. Prefer clips
    between min_seconds and max_seconds; fall back to the first clip.
    """
    for path in ref_audio_paths:
        try:
            info = sf.info(path)
            dur = info.frames / float(info.samplerate)
            if min_seconds <= dur <= max_seconds:
                return path
        except Exception:
            continue
    return ref_audio_paths[0]


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

def text_to_tokens(text, phonemizer, cleaner, device):
    """Phonemize text and convert to token tensor."""
    ps = phonemizer.phonemize([text.strip()])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)
    tokens = cleaner(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
    return tokens


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def inference(text_tokens, ref_s, model, sampler, model_params, device,
              alpha=0.3, beta=0.7, diffusion_steps=10, embedding_scale=1):
    """
    Synthesise speech.

    ref_s: [1, 256] — concatenation of [speaker_embedding (128) | prosodic (128)]
    alpha: 0=use ref acoustic directly, 1=use diffusion-predicted acoustic
    beta:  0=use ref prosodic directly, 1=use diffusion-predicted prosodic
    """
    with torch.no_grad():
        input_lengths = torch.LongTensor([text_tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(text_tokens, input_lengths, text_mask)
        bert_dur = model.bert(text_tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        s_pred = sampler(
            noise=torch.randn((1, 256)).unsqueeze(1).to(device),
            embedding=bert_dur,
            embedding_scale=embedding_scale,
            features=ref_s,
            num_steps=diffusion_steps,
        ).squeeze(1)  # [1, 256]

        # Blend diffusion output with reference
        ref = alpha * s_pred[:, :128] + (1 - alpha) * ref_s[:, :128]   # acoustic
        s = beta * s_pred[:, 128:] + (1 - beta) * ref_s[:, 128:]       # prosodic

        # Duration prediction
        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        # Alignment matrix
        pred_aln_trg = torch.zeros(input_lengths.item(), int(pred_dur.sum().item()))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].item())] = 1
            c_frame += int(pred_dur[i].item())

        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

    # Trim the trailing pulse artifact
    return out.squeeze().cpu().numpy()[..., :-50]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_styletts2(config, checkpoint_path, device):
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)

    from Utils.PLBERT.util import load_plbert
    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)

    model_params = recursive_munch(config['model_params'])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].to(device) for key in model]

    # Load checkpoint — handle DataParallel 'module.' prefix if present
    params_whole = torch.load(checkpoint_path, map_location='cpu')
    params = params_whole['net'] if 'net' in params_whole else params_whole

    for key in model:
        if key in params:
            state_dict = params[key]
            if any(k.startswith('module.') for k in state_dict.keys()):
                from collections import OrderedDict
                state_dict = OrderedDict(
                    (k[7:], v) for k, v in state_dict.items()
                )
            model[key].load_state_dict(state_dict, strict=False)
            print(f'  loaded: {key}')

    _ = [model[key].eval() for key in model]
    return model, model_params


def load_speaker_encoder(checkpoint_path, config_path, device):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    mc = cfg['speaker_encoder_model']
    encoder = SpeakerEncoder(
        mel_dim=mc['mel_dim'],
        hidden_dim=mc['hidden_dim'],
        attn_dim=mc['attn_dim'],
        embedding_out_dim=mc['embedding_out_dim'],
        N_prenet=mc['N_prenet'],
        N_conv=mc['N_conv'],
    ).to(device)
    encoder.load_state_dict(torch.load(checkpoint_path, map_location=device))
    encoder.eval()
    return encoder


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='StyleTTS2 + custom speaker embedding inference')
    parser.add_argument('--text', required=True,
                        help='Text to synthesize')
    parser.add_argument('--ref_audio_dir', required=True,
                        help='Directory of .wav files from the target speaker. '
                             'All wavs (recursive) are averaged into one embedding.')
    parser.add_argument('--checkpoint', default=DEFAULT_STYLETTS_CKPT,
                        help='Fine-tuned StyleTTS2 checkpoint (.pth). '
                             f'Default: {DEFAULT_STYLETTS_CKPT}')
    parser.add_argument('--config', default=DEFAULT_CONFIG,
                        help=f'StyleTTS2 config file. Default: {DEFAULT_CONFIG}')
    parser.add_argument('--speaker_encoder_checkpoint', default=DEFAULT_SPKENC_CKPT,
                        help='Trained speaker encoder checkpoint (.pt). '
                             f'Default: {DEFAULT_SPKENC_CKPT}')
    parser.add_argument('--speaker_encoder_config', default=DEFAULT_SPKENC_CONFIG,
                        help='Speaker encoder config yaml. '
                             f'Default: {DEFAULT_SPKENC_CONFIG}')
    parser.add_argument('--output', default='output.wav',
                        help='Output wav path (default: output.wav)')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='Acoustic style blend 0=ref, 1=diffusion (default 0.3)')
    parser.add_argument('--beta', type=float, default=0.7,
                        help='Prosodic style blend 0=ref, 1=diffusion (default 0.7)')
    parser.add_argument('--diffusion_steps', type=int, default=10,
                        help='Diffusion sampling steps (default 10)')
    parser.add_argument('--max_ref_clips', type=int, default=0,
                        help='If >0, cap the number of ref clips used for the '
                             'speaker embedding. 0 means use all (default).')
    args = parser.parse_args()

    # Existence checks with helpful error messages
    for label, path in [
        ('StyleTTS2 checkpoint', args.checkpoint),
        ('StyleTTS2 config', args.config),
        ('speaker encoder checkpoint', args.speaker_encoder_checkpoint),
        ('speaker encoder config', args.speaker_encoder_config),
    ]:
        if not osp.exists(path):
            raise FileNotFoundError(
                f'{label} not found at: {path}\n'
                f'Either place the file there or pass an explicit path flag.')

    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # Gather reference wavs
    ref_audio_paths = gather_wavs(args.ref_audio_dir)
    if args.max_ref_clips > 0 and len(ref_audio_paths) > args.max_ref_clips:
        print(f'Capping {len(ref_audio_paths)} clips to {args.max_ref_clips}')
        ref_audio_paths = ref_audio_paths[:args.max_ref_clips]
    print(f'Found {len(ref_audio_paths)} reference clip(s) in {args.ref_audio_dir}')

    # Load config
    config = yaml.safe_load(open(args.config))

    # Load models
    print(f'Loading StyleTTS2 from {args.checkpoint}...')
    model, model_params = load_styletts2(config, args.checkpoint, device)

    print(f'Loading speaker encoder from {args.speaker_encoder_checkpoint}...')
    speaker_encoder = load_speaker_encoder(
        args.speaker_encoder_checkpoint, args.speaker_encoder_config, device)

    # Diffusion sampler
    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False,
    )

    # Phonemizer
    import phonemizer
    global_phonemizer = phonemizer.backend.EspeakBackend(
        language='en-us', preserve_punctuation=True, with_stress=True)
    cleaner = TextCleaner()

    # Speaker embedding — average over every clip in the directory
    print(f'Computing averaged speaker embedding from {len(ref_audio_paths)} clip(s)...')
    speaker_emb = compute_speaker_embedding(ref_audio_paths, speaker_encoder, device)  # [1, 128]

    # Prosodic style — pick one clip to use as the prosodic reference
    prosody_clip = pick_prosody_reference(ref_audio_paths)
    print(f'Computing prosodic style from: {prosody_clip}')
    ref_p = compute_prosodic_style(prosody_clip, model, device)  # [1, 128]

    ref_s = torch.cat([speaker_emb, ref_p], dim=1)  # [1, 256]

    # Tokenize + synthesize
    tokens = text_to_tokens(args.text, global_phonemizer, cleaner, device)

    print('Synthesizing...')
    wav = inference(
        tokens, ref_s, model, sampler, model_params, device,
        alpha=args.alpha, beta=args.beta,
        diffusion_steps=args.diffusion_steps,
    )

    sf.write(args.output, wav, 24000)
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
