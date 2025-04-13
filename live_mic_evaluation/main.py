#!/usr/bin/env python3
"""
Live Microphone Evaluation with Speech Recognition
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

from audio_utils import AudioDeviceManager, AudioAnalyzer
from whisper_integration import WhisperManager

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Live Microphone Evaluation with Speech Recognition")
    
    parser.add_argument("--device", "-d", type=int, help="Audio input device ID (default: system default)")
    parser.add_argument("--list", "-l", action="store_true", help="List available audio input devices and exit")
    parser.add_argument("--duration", "-t", type=float, default=5.0, help="Recording duration in seconds (default: 5.0)")
    parser.add_argument("--sample-rate", "-sr", type=int, default=16000, help="Sample rate in Hz (default: 16000 for Whisper)")
    parser.add_argument("--channels", "-c", type=int, default=1, help="Number of channels (default: 1)")
    parser.add_argument("--output", "-o", type=str, help="Output directory for results (default: ./output)")
    parser.add_argument("--visualize", "-v", action="store_true", help="Generate audio visualizations")
    parser.add_argument("--skip-transcription", "-s", action="store_true", help="Skip speech transcription")
    parser.add_argument("--model", "-m", type=str, default="tiny.en", help="Whisper model to use (default: tiny.en)")
    
    return parser.parse_args()

def main():
    """Main function for live microphone evaluation."""
    args = parse_arguments()
    
    # List devices if requested
    if args.list:
        AudioDeviceManager.print_devices()
        return 0
    
    # Set up output directory
    output_dir = args.output or os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a timestamped directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Prepare file paths
    audio_file = os.path.join(run_dir, "recorded_audio.wav")
    report_file = os.path.join(run_dir, "evaluation_report.json")
    viz_file = os.path.join(run_dir, "audio_visualization.png") if args.visualize else None
    
    print(f"\n=== Live Microphone Evaluation ===")
    print(f"Output directory: {run_dir}")
    
    try:
        # Record audio
        print("\n1. Recording audio...")
        if args.device is not None:
            print(f"Using device ID: {args.device}")
        audio_data, sample_rate = AudioDeviceManager.record_audio(
            device_id=args.device,
            duration=args.duration,
            sample_rate=args.sample_rate,
            channels=args.channels
        )
        
        # Save the recorded audio
        saved_file = AudioDeviceManager.save_audio(audio_data, sample_rate, audio_file)
        print(f"Audio saved to: {saved_file}")
        
        # Analyze audio quality
        print("\n2. Analyzing audio quality...")
        metrics = AudioAnalyzer.compute_metrics(audio_data, sample_rate)
        print(f"RMS Level: {metrics['rms_level']:.4f}")
        print(f"Peak Level: {metrics['peak_level']:.4f}")
        print(f"Clipping: {metrics['clipping_percentage']:.2f}%")
        print(f"Silent segments: {metrics['silent_percentage']:.2f}%")
        if metrics.get('snr_db') is not None:
            print(f"Estimated SNR: {metrics['snr_db']:.2f} dB")
        
        # Generate visualizations if requested
        if args.visualize:
            print("\n3. Generating audio visualizations...")
            AudioAnalyzer.visualize_audio(audio_data, sample_rate, viz_file)
            print(f"Visualizations saved to: {viz_file}")
        
        # Prepare the report
        report = {
            "timestamp": timestamp,
            "audio_file": os.path.basename(audio_file),
            "audio_metrics": metrics,
        }
        
        # Run speech transcription if not skipped
        if not args.skip_transcription:
            print("\n4. Running speech recognition...")
            whisper_manager = WhisperManager(model_name=args.model)
            
            # Ensure Whisper.cpp and model are ready
            if not whisper_manager.ensure_ready():
                print("Failed to prepare Whisper.cpp for transcription.")
                report["transcription"] = {"error": "Whisper setup failed", "success": False}
            else:
                # Transcribe the audio
                transcription = whisper_manager.transcribe_audio(audio_file)
                
                if transcription["success"]:
                    print("\nTranscription result:")
                    if "text" in transcription:
                        print(f"{transcription['text']}")
                        report["transcription"] = {"text": transcription["text"], "success": True}
                    elif "transcript" in transcription:
                        print(f"{transcription['transcript']}")
                        report["transcription"] = {"text": transcription["transcript"], "success": True}
                    else:
                        print("No transcript found in output.")
                        report["transcription"] = {"error": "No transcript in output", "success": False}
                else:
                    print(f"Transcription failed: {transcription.get('error', 'Unknown error')}")
                    report["transcription"] = transcription
        
        # Save the report
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nEvaluation report saved to: {report_file}")
        print("=== Evaluation Complete ===\n")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 1
    except Exception as e:
        print(f"\nError: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())