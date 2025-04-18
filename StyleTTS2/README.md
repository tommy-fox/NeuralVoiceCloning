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

## Resources

- **Fine-Tuning Discussion**: [GitHub Discussions #100](https://github.com/yl4579/StyleTTS2/discussions/65)
  - **Fine-Tuned Samples**: 
    - [50 epochs config on rtx 4090; Dataset is madmonq interview sliced into 10 second segments](https://www.youtube.com/watch?v=Tuz7_7q0Pr0)
    - [Default config for 50 epochs and one hour of data]()
- **StyleTTS2 Repo**: [https://github.com/yl4579/StyleTTS2/](https://github.com/yl4579/StyleTTS2/tree/main)
  - **Paper**: [StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models](https://arxiv.org/abs/2306.07691)
  - **Audio Samples**: [https://styletts2.github.io/](https://styletts2.github.io/)
  - **Online Demo**: [Hugging Face](https://huggingface.co/spaces/styletts2/styletts2)