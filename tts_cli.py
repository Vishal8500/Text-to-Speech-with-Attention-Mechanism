import os
import torch
import argparse
from speechbrain.inference.TTS import Tacotron2
from speechbrain.inference.vocoders import HIFIGAN
import torchaudio
from speechbrain.utils.text_to_sequence import text_to_sequence

def synthesize_speech(text, output_path=None):
    """Synthesize speech from text using local models"""
    try:
        print("Loading Tacotron2...")
        # Use from_hparams instead of direct initialization
        tacotron2 = Tacotron2.from_hparams(
            source="speechbrain/tts-tacotron2-ljspeech",
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )
        
        print("Loading HiFiGAN...")
        hifi_gan = HIFIGAN.from_hparams(
            source="speechbrain/tts-hifigan-ljspeech",
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )

        # Generate output filename if not provided
        if output_path is None:
            safe_filename = "".join(x for x in text[:30] if x.isalnum() or x in (' ', '_'))
            safe_filename = safe_filename.replace(' ', '_')
            output_path = f"tts_outputs/{safe_filename}.wav"

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print(f"Converting text to speech: {text}")
        
        # Ensure text is properly formatted
        text = text.strip()
        if not text.endswith(('.', '!', '?')):
            text += '.'
            
        # Generate mel spectrogram
        with torch.no_grad():
            mel_output, mel_length, alignment = tacotron2.encode_text(text)
            
            # Convert to waveform
            waveforms = hifi_gan.decode_batch(mel_output)
            # Move to CPU and squeeze
            waveforms = waveforms.cpu().squeeze(1)
        
        # Save the audio file
        torchaudio.save(output_path, waveforms, sample_rate=22050)
        print(f"Audio saved to: {output_path}")
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True

    except Exception as e:
        print(f"Error during speech synthesis: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Text-to-Speech Conversion')
    parser.add_argument('text', type=str, help='Text to convert to speech')
    parser.add_argument('--output', '-o', type=str, help='Output WAV file path (optional)')
    args = parser.parse_args()
    synthesize_speech(args.text, args.output)

if __name__ == "__main__":
    main()