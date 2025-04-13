#!/usr/bin/env python3
"""
Simple wrapper for Whisper.cpp to transcribe audio files.
This wrapper handles the complexities of calling the Whisper.cpp executable.
"""
import os
import sys
import subprocess
import json
import argparse
from pathlib import Path
from datetime import datetime

# Default paths
WHISPER_EXECUTABLE = Path("build/whisper.cpp/build/bin/whisper-cli")
MODELS_DIR = Path("models")
DEFAULT_MODEL = "ggml-tiny.en.bin"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simple Whisper.cpp transcription wrapper")
    parser.add_argument("--audio", "-a", type=str, required=False,
                        help="Path to audio file to transcribe")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL,
                        help=f"Model file name (default: {DEFAULT_MODEL})")
    parser.add_argument("--language", "-l", type=str, default="en",
                        help="Language code (default: en)")
    parser.add_argument("--threads", "-t", type=int, default=4,
                        help="Number of threads to use (default: 4)")
    parser.add_argument("--beam-size", "-b", type=int, default=5,
                        help="Beam size for beam search (default: 5)")
    parser.add_argument("--record", "-r", action="store_true",
                        help="Record audio and transcribe (instead of using a file)")
    parser.add_argument("--duration", "-d", type=float, default=5.0,
                        help="Recording duration in seconds (default: 5.0 seconds)")
    parser.add_argument("--device", "-dev", type=int, default=1,
                        help="Audio device ID for recording (default: 1)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show verbose output")
    
    return parser.parse_args()

def ensure_model_exists(model_name):
    """Ensure the model file exists."""
    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        print(f"Error: Model file '{model_path}' not found.")
        print("Please download the model or check the path.")
        sys.exit(1)
    return model_path

def record_audio(duration=5.0, device_id=1):
    """Record audio from microphone."""
    try:
        import sounddevice as sd
        import scipy.io.wavfile as wav
        import numpy as np
        from tempfile import NamedTemporaryFile
        
        print(f"Recording {duration} seconds of audio from device {device_id}...")
        print("Speak clearly into your microphone...")
        
        # Start recording
        sample_rate = 16000
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.int16,
            device=device_id
        )
        
        # Display a progress bar
        for i in range(int(duration)):
            sys.stdout.write(f"\rProgress: [{'#' * (i+1)}{'.' * (int(duration)-i-1)}] {((i+1)/duration)*100:.1f}%")
            sys.stdout.flush()
            sd.sleep(1000)
            
        # Wait for recording to complete
        sd.wait()
        print("\nRecording complete!")
        
        # Create a temporary WAV file
        temp_file = NamedTemporaryFile(suffix=".wav", delete=False)
        wav_path = temp_file.name
        temp_file.close()
        
        # Save the audio
        wav.write(wav_path, sample_rate, audio_data)
        print(f"Audio saved to temporary file: {wav_path}")
        
        return wav_path
        
    except Exception as e:
        print(f"Error during recording: {e}")
        sys.exit(1)

def transcribe(audio_file, model_path, language="en", threads=4, beam_size=5, verbose=False):
    """Transcribe audio using Whisper.cpp."""
    # Check if the executable exists
    if not WHISPER_EXECUTABLE.exists():
        print(f"Error: Whisper executable not found at {WHISPER_EXECUTABLE}")
        print("Have you built Whisper.cpp?")
        sys.exit(1)
        
    # Check if audio file exists
    audio_path = Path(audio_file)
    if not audio_path.exists():
        print(f"Error: Audio file '{audio_path}' not found.")
        sys.exit(1)
    
    # Build command
    cmd = [
        str(WHISPER_EXECUTABLE),
        "-m", str(model_path),
        "-f", str(audio_path),
        "-l", language,
        "-t", str(threads),
        "--beam-size", str(beam_size)
    ]
    
    if verbose:
        print(f"Running command: {' '.join(cmd)}")
    
    # Execute transcription
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error during transcription: {result.stderr}")
            return None
            
        # Extract the transcription text from the output
        output_lines = result.stdout.strip().split('\n')
        transcription_lines = []
        
        # Find lines with timestamps [time] text
        for line in output_lines:
            if ']' in line and '[' in line:
                # This is a line with a timestamp
                transcription_lines.append(line)
                
        if not transcription_lines and verbose:
            print("Raw output:")
            print(result.stdout)
            
        return transcription_lines if transcription_lines else result.stdout
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def main():
    args = parse_args()
    
    # Ensure model exists
    model_path = ensure_model_exists(args.model)
    
    # Get audio file
    audio_file = None
    if args.record:
        audio_file = record_audio(duration=args.duration, device_id=args.device)
    elif args.audio:
        audio_file = args.audio
    else:
        print("Error: Either specify an audio file with --audio or use --record to record new audio.")
        sys.exit(1)
    
    # Transcribe
    print(f"\nTranscribing with {args.model} model...")
    
    results = transcribe(
        audio_file=audio_file,
        model_path=model_path,
        language=args.language,
        threads=args.threads,
        beam_size=args.beam_size,
        verbose=args.verbose
    )
    
    # Clean up temporary file if recording was used
    if args.record and audio_file and os.path.exists(audio_file):
        if args.verbose:
            print(f"Removing temporary file: {audio_file}")
        try:
            os.unlink(audio_file)
        except Exception as e:
            if args.verbose:
                print(f"Error removing temporary file: {e}")
    
    # Display results
    if results:
        print("\n===== TRANSCRIPTION RESULTS =====")
        if isinstance(results, list):
            for line in results:
                print(line)
        else:
            print(results)
    else:
        print("No transcription results.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())