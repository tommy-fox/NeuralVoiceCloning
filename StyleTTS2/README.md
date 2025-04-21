# Exploring Speaker-Style Adaptation through Fine-Tuning

This repository focuses on exploring speaker-style adaptation using StyleTTS2, a state-of-the-art text-to-speech (TTS) model. The goal is to fine-tune StyleTTS2 for specific voice cloning tasks, enabling it to adapt to new speakers or styles with minimal data.

## Goal

The primary purpose of this repository is to investigate and enhance the speaker-style adaptation capabilities of StyleTTS2 through fine-tuning. We will fine-tuning the StyleTTS2 base model on the following datasets:
- [Voxceleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)
- [VCTK](https://paperswithcode.com/dataset/vctk)

### Fine-Tuning Details

- The [LibriTTS](https://www.openslr.org/60) checkpoint will be used as the pre-trained model.
- The default configuration `config_ft.yml` fine-tunes on [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) with 1 hour of speech data (around 1k samples) for 50 epochs. This process takes approximately 4 hours on four NVIDIA A100 GPUs.
  - The fine-tuned model's quality is slightly worse (similar to NaturalSpeech on LJSpeech) than a model trained from scratch on 24 hours of speech data, which takes around 2.5 days on four A100 GPUs.
- See notes on Fine-Tuning:
  - https://github.com/yl4579/StyleTTS2/discussions/81
  - https://github.com/IIEleven11/StyleTTS2FineTune

## Resources

- **Model Checkpoints**:
  - **[StyleTTS2-LibreTTS](https://huggingface.co/yl4579/StyleTTS2-LibriTTS/tree/main) (multi-speaker)**:
    - StyleTTS2-LibreTTS is a pre-trained TTS model trained on the LibreTTS corpus. [LibriTTS](https://www.openslr.org/60) is a **multi-speaker** English corpus of approximately 585 hours of read English speech at 24kHz sampling rate, prepared by Heiga Zen with the assistance of Google Speech and Google Brain team members. Designed for TTS research, it is derived from the original materials of the LibriSpeech corpus.
    - **Model checkpoint** is accessible via [this link](https://huggingface.co/yl4579/StyleTTS2-LibriTTS/tree/main/Models/LibriTTS)
  - **[StyleTTS2-LJSpeech](https://huggingface.co/yl4579/StyleTTS2-LJSpeech/tree/main) (single speaker)**:
    - StyleTTS2-LJSpeech is a pre-trained TTS model trained on the LJSpeech dataset. The [LJSpeech](https://www.openslr.org/60) dataset is a public domain speech dataset consisting of 13,100 short audio clips of a **single speaker** reading passages from 7 non-fiction books. A transcription is provided for each clip. Clips vary in length from 1 to 10 seconds and have a total length of approximately 24 hours.
    - **Model checkpoint** is accessible via [this link](https://huggingface.co/yl4579/StyleTTS2-LJSpeech/tree/main)
  - **[Kokoro 82M](https://github.com/hexgrad/kokoro?tab=readme-ov-file)**:
    - Kokoro is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, Kokoro can be deployed anywhere from production environments to personal projects.
    - **Model checkpoint** is accessible via [this link](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/kokoro-v1_0.pth)
    - **Voices** are accessible via [this link](https://huggingface.co/hexgrad/Kokoro-82M/tree/main/voices)
  
- **Training Datasets**:
  - **[LibriTTS](https://www.openslr.org/60)**
  - **[LJSpeech](https://www.openslr.org/60)**
  
- **StyleTTS2 Repo**: [https://github.com/yl4579/StyleTTS2/](https://github.com/yl4579/StyleTTS2/tree/main)
  - **Paper**: [StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models](https://arxiv.org/abs/2306.07691)
  - **Audio Samples**: [https://styletts2.github.io/](https://styletts2.github.io/)
  - **Online Demo**: [Hugging Face](https://huggingface.co/spaces/styletts2/styletts2)