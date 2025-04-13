#!/usr/bin/env python3
"""
Continuous Live Transcription using Whisper.cpp

This script records audio from the microphone in chunks and transcribes
each chunk in real-time, providing a continuously updating transcript.
"""
import os
import sys
import time
import tempfile
import argparse
import threading
import queue
import signal
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from datetime import datetime
from pathlib import Path

# Add color support
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Default paths
WHISPER_EXECUTABLE = Path("build/whisper.cpp/build/bin/whisper-cli")
MODELS_DIR = Path("models")
DEFAULT_MODEL = "ggml-tiny.en.bin"

# Global variables for clean exit
running = True
audio_queue = queue.Queue()
transcription_history = []
temp_dir = None
transcriber_thread = None

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global running
    print(f"\n{Colors.YELLOW}Stopping transcription...{Colors.ENDC}")
    running = False
    
    # Allow time for threads to clean up
    if transcriber_thread and transcriber_thread.is_alive():
        transcriber_thread.join(timeout=2.0)
    
    # Save the full transcript
    save_transcript()
    
    # Clean up temp directory
    clean_up()
    
    sys.exit(0)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Continuous Live Transcription with Whisper.cpp")
    parser.add_argument("--device", "-d", type=int, default=1,
                        help="Audio device ID (default: 1)")
    parser.add_argument("--chunk", "-c", type=float, default=3.0,
                        help="Length of recording chunks in seconds (default: 3.0)")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL,
                        help=f"Model file name (default: {DEFAULT_MODEL})")
    parser.add_argument("--language", "-l", type=str, default="en",
                        help="Language code (default: en)")
    parser.add_argument("--threads", "-t", type=int, default=4,
                        help="Number of threads to use for transcription (default: 4)")
    parser.add_argument("--list-devices", "-ld", action="store_true",
                        help="List available audio devices and exit")
    parser.add_argument("--save-audio", "-sa", action="store_true",
                        help="Save all audio chunks to disk")
    return parser.parse_args()

def list_audio_devices():
    """List available audio input devices"""
    print(f"{Colors.HEADER}Available audio input devices:{Colors.ENDC}")
    print("-" * 60)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"ID: {i}, {Colors.BOLD}Name: {device['name']}{Colors.ENDC}, "
                  f"Channels: {device['max_input_channels']}, "
                  f"Sample Rate: {device['default_samplerate']}")
    print("-" * 60)

def ensure_model_exists(model_path):
    """Check if the model file exists"""
    if not model_path.exists():
        print(f"{Colors.RED}Error: Model file not found at {model_path}{Colors.ENDC}")
        print("Please download the model or ensure the path is correct.")
        sys.exit(1)

def ensure_whisper_executable():
    """Check if the Whisper.cpp executable exists"""
    if not WHISPER_EXECUTABLE.exists():
        print(f"{Colors.RED}Error: Whisper executable not found at {WHISPER_EXECUTABLE}{Colors.ENDC}")
        print("Please ensure Whisper.cpp is correctly built.")
        sys.exit(1)

def record_audio_chunk(device_id, duration, sample_rate=16000, chunk_num=None):
    """Record a chunk of audio"""
    global running
    
    # Create identifier for this chunk
    timestamp = datetime.now().strftime("%H:%M:%S")
    chunk_id = f"chunk_{chunk_num}" if chunk_num is not None else "chunk"
    
    # Start recording
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype=np.int16,
        device=device_id
    )
    
    # Display progress bar
    for i in range(int(duration)):
        if not running:
            break
        progress = int((i + 1) / duration * 30)
        bar = f"[{'#' * progress}{'-' * (30 - progress)}]"
        percent = ((i + 1) / duration) * 100
        sys.stdout.write(f"\r{Colors.GREEN}Recording {chunk_id}: {bar} {percent:.1f}%{Colors.ENDC}")
        sys.stdout.flush()
        time.sleep(1)
    
    # Wait for recording to complete
    sd.wait()
    print()  # New line after progress bar
    
    # Create temp file path
    global temp_dir
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="whisper_live_")
    
    file_name = f"{chunk_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    output_path = os.path.join(temp_dir, file_name)
    
    # Save audio to WAV file
    wav.write(output_path, sample_rate, audio_data)
    
    # If save_audio flag is set, also save to output directory
    global args
    if args.save_audio:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        save_path = output_dir / file_name
        wav.write(save_path, sample_rate, audio_data)
    
    return output_path

def transcribe_audio(audio_file, model_path, language="en", threads=4, beam_size=5):
    """Transcribe audio file using Whisper.cpp"""
    import subprocess
    
    # Build command
    cmd = [
        str(WHISPER_EXECUTABLE),
        "-m", str(model_path),
        "-f", audio_file,
        "-l", language,
        "-t", str(threads),
        "--beam-size", str(beam_size)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"{Colors.RED}Transcription error: {result.stderr}{Colors.ENDC}")
            return ""
        
        # Extract transcription from output
        output_lines = result.stdout.strip().split('\n')
        transcription = ""
        
        # Find lines with timestamps [time] text
        for line in output_lines:
            if ']' in line and '[' in line and len(line) > 5:  # Avoid empty timestamps
                # Extract just the text without timestamps if desired
                # text = line.split(']', 1)[1].strip()
                transcription += line + "\n"
        
        return transcription.strip()
        
    except Exception as e:
        print(f"{Colors.RED}Error during transcription: {e}{Colors.ENDC}")
        return ""

def transcriber_worker():
    """Worker thread to process audio files from the queue"""
    global running, audio_queue, transcription_history, args
    
    model_path = MODELS_DIR / args.model
    
    while running:
        try:
            # Get an item from the queue with timeout
            audio_file = audio_queue.get(timeout=1.0)
            
            # Transcribe the audio file
            print(f"{Colors.BLUE}Transcribing chunk...{Colors.ENDC}")
            transcript = transcribe_audio(
                audio_file=audio_file,
                model_path=model_path,
                language=args.language,
                threads=args.threads
            )
            
            # Add to history if we got a transcript
            if transcript:
                transcription_history.append(transcript)
                
                # Print the transcript with timestamp
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"\n{Colors.HEADER}[{timestamp}] TRANSCRIPT:{Colors.ENDC}")
                print(f"{Colors.YELLOW}{transcript}{Colors.ENDC}")
                print("-" * 60)
            else:
                print(f"{Colors.YELLOW}No transcription for this chunk.{Colors.ENDC}")
            
            # Delete temp file if not saving
            if not args.save_audio and os.path.exists(audio_file) and audio_file.startswith(temp_dir):
                try:
                    os.unlink(audio_file)
                except:
                    pass
            
            # Mark item as done
            audio_queue.task_done()
            
        except queue.Empty:
            # No items in queue, just continue
            continue
        except Exception as e:
            print(f"{Colors.RED}Error in transcriber worker: {e}{Colors.ENDC}")

def save_transcript():
    """Save the complete transcript to a file"""
    global transcription_history
    
    if not transcription_history:
        print(f"{Colors.YELLOW}No transcripts to save.{Colors.ENDC}")
        return
    
    try:
        # Prepare output directory and file
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        transcript_file = output_dir / f"full_transcript_{timestamp}.txt"
        
        # Write all transcripts to file
        with open(transcript_file, 'w') as f:
            f.write("\n\n".join(transcription_history))
        
        print(f"{Colors.GREEN}Full transcript saved to: {transcript_file}{Colors.ENDC}")
        
    except Exception as e:
        print(f"{Colors.RED}Error saving transcript: {e}{Colors.ENDC}")

def clean_up():
    """Clean up temporary files"""
    global temp_dir
    
    if temp_dir and os.path.exists(temp_dir):
        try:
            import shutil
            shutil.rmtree(temp_dir)
            print(f"{Colors.BLUE}Cleaned up temporary files.{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.YELLOW}Error cleaning up temp files: {e}{Colors.ENDC}")

def main():
    global running, audio_queue, transcription_history, args, transcriber_thread
    
    # Set up signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse arguments
    args = parse_args()
    
    # List devices if requested
    if args.list_devices:
        list_audio_devices()
        return 0
    
    # Check if model exists
    model_path = MODELS_DIR / args.model
    ensure_model_exists(model_path)
    
    # Check if whisper executable exists
    ensure_whisper_executable()
    
    # Start the transcriber thread
    transcriber_thread = threading.Thread(target=transcriber_worker, daemon=True)
    transcriber_thread.start()
    
    # Print header
    print(f"\n{Colors.HEADER}{'=' * 30} LIVE TRANSCRIPTION {'=' * 30}{Colors.ENDC}")
    print(f"{Colors.BOLD}Using model: {args.model}{Colors.ENDC}")
    print(f"Device: {args.device}, Chunk duration: {args.chunk}s, Language: {args.language}")
    print(f"{Colors.YELLOW}Press Ctrl+C to stop recording and save the transcript{Colors.ENDC}")
    print("=" * 80)
    
    # Main recording loop
    chunk_num = 1
    
    try:
        while running:
            # Record audio chunk
            audio_file = record_audio_chunk(
                device_id=args.device,
                duration=args.chunk,
                chunk_num=chunk_num
            )
            
            # Add to queue for transcription
            audio_queue.put(audio_file)
            
            # Increment chunk number
            chunk_num += 1
            
    except KeyboardInterrupt:
        # Handle Ctrl+C (should be caught by signal handler)
        pass
    finally:
        # Make sure we clean up
        running = False
        if transcriber_thread and transcriber_thread.is_alive():
            transcriber_thread.join(timeout=2.0)
        save_transcript()
        clean_up()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())