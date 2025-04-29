
# Text-to-Speech (TTS) System using Tacotron2 and HiFi-GAN

## Overview

This project is a complete Text-to-Speech (TTS) pipeline built using deep learning models **Tacotron2** and **HiFi-GAN**, both sourced from the [SpeechBrain](https://speechbrain.readthedocs.io/) library. Our system converts raw text inputs into high-quality, natural-sounding human speech.

---

## Key Components

- **Tacotron2**: An attention-based sequence-to-sequence model that converts input text into mel-spectrograms.
- **HiFi-GAN**: A high-fidelity neural vocoder that transforms mel-spectrograms into realistic audio waveforms.

---

## Workflow

1. **Text Tokenization**
   - Input text is converted into a sequence of numerical tokens to be processed by the model.

2. **Mel-Spectrogram Generation with Tacotron2**
   - Tacotron2 encodes the input token sequence and uses an **attention mechanism** to generate a mel-spectrogram aligned with the text.
   - Attention allows the model to focus on the correct text parts while generating audio frames.

3. **Waveform Synthesis with HiFi-GAN**
   - The mel-spectrogram output is passed to HiFi-GAN to synthesize a realistic speech waveform.
   - HiFi-GAN produces high-quality and intelligible audio efficiently.

4. **Training Dataset**
   - The **LJSpeech** dataset is used for training and evaluation.
   - Contains thousands of English audio clips and corresponding transcripts for robust training.

---

## Features

- End-to-end TTS system: from raw text to human-like speech.
- Attention-based alignment for better pronunciation and intonation.
- High-fidelity audio output using HiFi-GAN vocoder.
- Scalable and extendable using the modular architecture of SpeechBrain.

---

## Dataset

- **LJSpeech Dataset**
  - 13,100 short audio clips of a single speaker reading passages from 7 non-fiction books.
  - Publicly available and widely used for speech synthesis tasks.

---

## Technologies Used

- Python
- PyTorch
- SpeechBrain
- Tacotron2
- HiFi-GAN
- LJSpeech Dataset

---

## Installation

```bash
git clone hhttps://github.com/Vishal8500/Text-to-Speech-with-Attention-Mechanism
cd Text-to-Speech-with-Attention-Mechanism
pip install -r requirements.txt
```

---

## Usage

```python
# Run inference script
python tts_cli.py "Abishek and Vishal played football together after school. They laughed and cheered as they scored goals against their friends."  
```

The output will be saved as a `.wav` file in the `tts_outputs/` directory.

---

## Credits

- SpeechBrain Team
- NVIDIA for Tacotron2 and HiFi-GAN
- [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/)

---

## Contributors

- [Vishal](https://github.com/Vishal8500) - Developer

---

## Future Enhancements

- Multi-speaker and multilingual TTS support
- Real-time streaming TTS
- Web interface integration

---
