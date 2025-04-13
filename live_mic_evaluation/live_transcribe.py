#!/usr/bin/env python3
"""
Live Transcription with Whisper.cpp

This script provides real-time transcription from a microphone input
using Whisper.cpp's tiny.en model. It captures audio in short segments
and provides continuous transcription.
"""
import os
import sys
import time
import argparse
import queue
import tempfile
import threading
import traceback
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
from datetime import datetime

from audio_utils import AudioDeviceManager, AudioAnalyzer
from whisper_integration import WhisperManager

# Global variables
audio_queue = queue.Queue()
is_running = True
transcript_buffer = []
last_transcript_time = time.time()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Live Microphone Transcription")
    
    parser.add_argument("--device", "-d", type=int, help="Audio input device ID (default: system default)")
    parser.add_argument("--list", "-l", action="store_true", help="List available audio input devices and exit")
    parser.add_argument("--sample-rate", "-sr", type=int, default=16000, help="Sample rate in Hz (default: 16000 for Whisper)")
    parser.add_argument("--segment-length", "-sl", type=float, default=3.0, help="Audio segment length in seconds (default: 3.0)")
    parser.add_argument("--model", "-m", type=str, default="tiny.en", help="Whisper model to use (default: tiny.en)")
    parser.add_argument("--save-audio", "-s", action="store_true", help="Save recorded audio segments")
    parser.add_argument("--output", "-o", type=str, help="Output directory for results (default: ./output/live_session)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with detailed logging")
    
    return parser.parse_args()

def audio_callback(indata, frames, time_info, status):
    """Callback for audio input stream."""
    if status:
        print(f"Status: {status}")
    # Add the audio data to the queue
    audio_queue.put(indata.copy())

def save_audio_segment(audio_data, sample_rate, output_dir, segment_num):
    """Save an audio segment to disk."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = os.path.join(output_dir, f"segment_{segment_num:04d}.wav")
    wavfile.write(filename, sample_rate, audio_data)
    return filename

def transcribe_worker(whisper_manager, sample_rate, segment_length, save_audio=False, output_dir=None, debug=False):
    """Worker thread to process audio segments and transcribe them."""
    global is_running, transcript_buffer, last_transcript_time
    
    # Initialize counters
    segment_num = 0
    silence_counter = 0
    
    print("\n[Live Transcription Started]")
    print("Speak into your microphone. Press Ctrl+C to stop.\n")
    print("-" * 60)
    
    # Main processing loop
    while is_running:
        try:
            # Get audio data from queue with timeout
            audio_data = audio_queue.get(timeout=1.0)
            
            if debug:
                print(f"Processing audio segment {segment_num}, shape: {audio_data.shape}")
            
            # Convert from float32 to int16 format
            audio_data_int16 = np.int16(audio_data * 32767)
            
            # Check if audio is mostly silence to avoid unnecessary processing
            audio_energy = np.sqrt(np.mean(audio_data**2))
            if debug:
                print(f"Audio energy: {audio_energy:.6f}")
                
            if audio_energy < 0.01:  # Threshold can be adjusted
                silence_counter += 1
                if silence_counter > 2:  # Skip if multiple silent segments in a row
                    if debug:
                        print(f"Skipping silent segment {segment_num}")
                    audio_queue.task_done()
                    continue
            else:
                silence_counter = 0  # Reset silence counter if audio detected
            
            # Create a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_filename = temp_file.name
                if debug:
                    print(f"Saving temporary file: {temp_filename}")
                wavfile.write(temp_filename, sample_rate, audio_data_int16)
            
            # Save audio segment if requested
            if save_audio and output_dir:
                saved_file = save_audio_segment(audio_data_int16, sample_rate, 
                                              os.path.join(output_dir, "audio_segments"), 
                                              segment_num)
                if debug:
                    print(f"Saved audio segment: {saved_file}")
                
            # Transcribe the audio segment
            if debug:
                print(f"Transcribing segment {segment_num}...")
                
            result = whisper_manager.transcribe_audio(temp_filename)
            
            # Clean up temporary file
            try:
                os.unlink(temp_filename)
            except Exception as e:
                if debug:
                    print(f"Error deleting temp file: {e}")
            
            # Process and display the transcription
            if result["success"]:
                transcript = ""
                if "text" in result:
                    transcript = result["text"].strip()
                elif "transcript" in result:
                    transcript = result["transcript"].strip()
                
                if debug:
                    print(f"Raw transcript: '{transcript}'")
                    
                if transcript and not transcript.isspace():
                    # Add timestamp and print
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    
                    # Update transcript buffer
                    transcript_buffer.append(transcript)
                    if len(transcript_buffer) > 5:  # Keep only the last 5 segments
                        transcript_buffer.pop(0)
                    
                    # Print the new transcript with timestamp
                    print(f"[{timestamp}] {transcript}")
                    last_transcript_time = time.time()
                    
                    # Every few segments, show the combined recent transcript
                    if segment_num % 5 == 0 and segment_num > 0:
                        print("\n" + "-" * 60)
                        print("RECENT CONVERSATION:")
                        print(" ".join(transcript_buffer))
                        print("-" * 60 + "\n")
            else:
                if debug:
                    print(f"Transcription failed: {result.get('error', 'Unknown error')}")
            
            segment_num += 1
            audio_queue.task_done()
            
        except queue.Empty:
            # Queue is empty, just continue
            pass
        except Exception as e:
            print(f"Error in transcription worker: {e}")
            if debug:
                traceback.print_exc()
            audio_queue.task_done()
    
    print("\n[Live Transcription Stopped]")

def main():
    """Main function for live transcription."""
    global is_running
    
    args = parse_arguments()
    
    debug_mode = args.debug
    if debug_mode:
        print("Debug mode enabled - detailed logging will be shown")
    
    # List devices if requested
    if args.list:
        AudioDeviceManager.print_devices()
        return 0
    
    # Set up output directory if saving audio
    output_dir = None
    if args.save_audio or args.output:
        output_dir = args.output or os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                              "output", "live_session")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_dir, f"session_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
    
    # Initialize Whisper manager
    print("Initializing Whisper.cpp...")
    whisper_manager = WhisperManager(model_name=args.model)
    
    # Ensure Whisper is ready (this will download the model if needed)
    if not whisper_manager.ensure_ready():
        print("Failed to initialize Whisper. Check the model and try again.")
        return 1
    
    # Show the selected device
    if args.device is not None:
        print(f"Using audio device ID: {args.device}")
    
    try:
        # Create a processing thread for transcription
        worker_thread = threading.Thread(
            target=transcribe_worker,
            args=(whisper_manager, args.sample_rate, args.segment_length, 
                  args.save_audio, output_dir, debug_mode)
        )
        worker_thread.daemon = True
        worker_thread.start()
        
        # Calculate buffer length based on segment length (in samples)
        buffer_length = int(args.sample_rate * args.segment_length)
        
        print("Starting audio stream...")
        
        # Start audio input stream
        with sd.InputStream(
            callback=audio_callback,
            device=args.device,
            channels=1,
            samplerate=args.sample_rate,
            blocksize=buffer_length
        ):
            print(f"Listening on device {args.device if args.device is not None else 'default'}...")
            
            # Keep the main thread running to handle Ctrl+C
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nStopping transcription...")
                is_running = False
                time.sleep(1)  # Give worker thread time to finish
                
    except Exception as e:
        print(f"Error: {e}")
        if debug_mode:
            traceback.print_exc()
        is_running = False
    
    return 0

if __name__ == "__main__":
    sys.exit(main())