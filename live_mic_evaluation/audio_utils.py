"""
Audio utilities for microphone device management and recording.
"""
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import librosa
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import os
import json
import time
import sys

class AudioDeviceManager:
    """Manages audio devices and recording capabilities."""
    
    @staticmethod
    def list_devices() -> List[Dict]:
        """List all available audio input devices."""
        devices = []
        try:
            for i, device in enumerate(sd.query_devices()):
                if device['max_input_channels'] > 0:
                    devices.append({
                        'id': i,
                        'name': device['name'],
                        'channels': device['max_input_channels'],
                        'default_samplerate': device['default_samplerate']
                    })
            if not devices:
                print("Warning: No audio input devices detected.")
        except Exception as e:
            print(f"Error listing audio devices: {e}")
        return devices
    
    @staticmethod
    def print_devices():
        """Print all available audio input devices."""
        devices = AudioDeviceManager.list_devices()
        if devices:
            print("Available audio input devices:")
            for device in devices:
                print(f"ID: {device['id']}, Name: {device['name']}, Channels: {device['channels']}")
        else:
            print("No audio input devices found.")
            print("Please check your microphone connections and system permissions.")
    
    @staticmethod
    def record_audio(
        device_id: Optional[int] = None,
        duration: float = 5.0,
        sample_rate: int = 16000,
        channels: int = 1
    ) -> Tuple[np.ndarray, int]:
        """
        Record audio from the specified device.
        
        Args:
            device_id: ID of the input device (None for default)
            duration: Recording duration in seconds
            sample_rate: Sample rate in Hz
            channels: Number of channels (1 for mono, 2 for stereo)
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Check if device exists
            if device_id is not None:
                devices = AudioDeviceManager.list_devices()
                device_ids = [d['id'] for d in devices]
                if device_id not in device_ids:
                    print(f"Warning: Device ID {device_id} not found. Using default device.")
                    device_id = None
            
            # Start recording
            print(f"Recording audio for {duration} seconds...")
            print("Speak now...")
            
            # Add a short delay to make sure message is printed before recording starts
            time.sleep(0.1)
            
            audio_data = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=channels,
                dtype='float32',
                device=device_id
            )
            
            # Show a simple countdown
            for i in range(int(duration), 0, -1):
                time.sleep(1)
                sys.stdout.write(f"\rRecording: {i} seconds left...")
                sys.stdout.flush()
            
            sd.wait()  # Wait until recording is finished
            print("\nRecording finished.")
            return audio_data, sample_rate
            
        except Exception as e:
            print(f"Error recording audio: {e}")
            print("Please check your microphone connections and system permissions.")
            # Return a short empty array so the program can continue
            return np.zeros((int(sample_rate * duration), channels)), sample_rate
    
    @staticmethod
    def save_audio(
        audio_data: np.ndarray,
        sample_rate: int,
        output_file: str
    ) -> str:
        """
        Save audio data to a WAV file.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate in Hz
            output_file: Path to output WAV file
            
        Returns:
            Path to the saved file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Normalize audio data to prevent clipping
        audio_data_norm = np.int16(audio_data * 32767)
        wav.write(output_file, sample_rate, audio_data_norm)
        return output_file


class AudioAnalyzer:
    """Analyzes audio data for quality metrics."""
    
    @staticmethod
    def compute_metrics(audio_data: np.ndarray, sample_rate: int) -> Dict:
        """
        Compute audio quality metrics.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate in Hz
            
        Returns:
            Dictionary of audio metrics
        """
        # Convert to mono if stereo
        if audio_data.ndim > 1 and audio_data.shape[1] > 1:
            audio_mono = np.mean(audio_data, axis=1)
        else:
            audio_mono = audio_data.flatten()
            
        # Calculate metrics
        metrics = {
            'rms_level': float(np.sqrt(np.mean(audio_mono**2))),
            'peak_level': float(np.max(np.abs(audio_mono))),
            'clipping_percentage': float(np.mean(np.abs(audio_mono) > 0.95) * 100),
            'length_seconds': float(len(audio_mono) / sample_rate),
            'silent_percentage': float(np.mean(np.abs(audio_mono) < 0.01) * 100),
            'sample_rate': sample_rate
        }
        
        # Calculate Signal-to-Noise Ratio (SNR) if possible
        try:
            # Simple SNR estimation: assume first 0.5s is mostly noise
            noise_segment = audio_mono[:int(0.5 * sample_rate)]
            noise_power = np.mean(noise_segment**2)
            signal_power = np.mean(audio_mono**2)
            if noise_power > 0:
                metrics['snr_db'] = float(10 * np.log10(signal_power / noise_power))
            else:
                metrics['snr_db'] = float('inf')
        except Exception:
            metrics['snr_db'] = None
            
        return metrics
    
    @staticmethod
    def visualize_audio(
        audio_data: np.ndarray, 
        sample_rate: int, 
        output_file: Optional[str] = None
    ):
        """
        Generate visualizations of the audio data.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate in Hz
            output_file: Path to save the visualization, if None just displays
        """
        # Convert to mono if stereo
        if audio_data.ndim > 1 and audio_data.shape[1] > 1:
            audio_mono = np.mean(audio_data, axis=1)
        else:
            audio_mono = audio_data.flatten()
        
        # Create figure with two subplots
        plt.figure(figsize=(12, 10))
        
        # Plot waveform
        plt.subplot(3, 1, 1)
        plt.title('Waveform')
        time_axis = np.linspace(0, len(audio_mono)/sample_rate, len(audio_mono))
        plt.plot(time_axis, audio_mono)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        
        # Plot spectrogram
        plt.subplot(3, 1, 2)
        plt.title('Spectrogram')
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_mono)), ref=np.max)
        librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        
        # Plot volume levels over time
        plt.subplot(3, 1, 3)
        plt.title('Volume Level')
        frame_length = int(sample_rate * 0.025)  # 25ms frames
        hop_length = int(sample_rate * 0.010)    # 10ms hop
        
        # Calculate RMS energy in dB
        rms = librosa.feature.rms(y=audio_mono, frame_length=frame_length, hop_length=hop_length)[0]
        db_rms = 20 * np.log10(rms + 1e-6)  # Convert to dB, avoid log(0)
        
        frames_time = librosa.frames_to_time(np.arange(len(rms)), sr=sample_rate, hop_length=hop_length)
        plt.plot(frames_time, db_rms)
        plt.axhline(y=-20, color='r', linestyle='--', alpha=0.7, label='Good signal (-20dB)')
        plt.axhline(y=-60, color='y', linestyle='--', alpha=0.7, label='Low signal (-60dB)')
        plt.xlabel('Time (s)')
        plt.ylabel('dB')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if output_file:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            plt.savefig(output_file)
        else:
            plt.show()
        
        plt.close()