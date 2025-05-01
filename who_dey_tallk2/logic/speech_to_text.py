#!/usr/bin/env python3
"""
Speech-to-Text module for Who Dey Tallk 2

Handles audio recording and speech transcription
"""
import os
import sys
import time
import logging
import threading
import tempfile
import subprocess
import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
import queue # Add queue import

logger = logging.getLogger(__name__)

class SpeechToText:
    """
    Speech-to-text transcription class
    
    Records audio and converts speech to text using Whisper or other models.
    """
    
    def __init__(self, model_name="tiny.en", language="en", sample_rate=16000):
        """
        Initialize the speech-to-text system
        
        Args:
            model_name: Name of the model to use (tiny.en, base.en, etc.)
            language: Language code for transcription
            sample_rate: Sample rate for audio recording
        """
        self.model_name = model_name
        self.language = language
        self.sample_rate = sample_rate
        
        # Internal state
        self.whisper_cli = None
        self.model_path = None
        self.base_dir = Path(__file__).parent.parent.resolve()
        self.temp_dir = None
        self.lock = threading.RLock()
        self.stream = None
        self.audio_queue = queue.Queue()
        self.stream_stop_event = threading.Event() # Event to signal stream stop
        
        # Initialize the transcription system
        self._init_system()
    
    def _init_system(self):
        """Initialize the speech-to-text system"""
        try:
            # Find the Whisper CLI executable
            whisper_paths = [
                self.base_dir / "build" / "whisper.cpp" / "build" / "bin" / "whisper-cli",
                self.base_dir / "build" / "whisper" / "whisper",
                self.base_dir.parent / "live_mic_evaluation" / "build" / "whisper.cpp" / "build" / "bin" / "whisper-cli"
            ]
            
            for path in whisper_paths:
                if path.exists() and os.access(path, os.X_OK):
                    self.whisper_cli = path
                    logger.info(f"Found Whisper CLI at {path}")
                    break
                    
            if not self.whisper_cli:
                logger.warning("Could not find Whisper CLI executable")
                
            # Find the model file
            model_paths = [
                self.base_dir / "models" / f"ggml-{self.model_name}.bin",
                self.base_dir.parent / "live_mic_evaluation" / "models" / f"ggml-{self.model_name}.bin",
                self.base_dir.parent / "models" / f"ggml-{self.model_name}.bin"
            ]
            
            for path in model_paths:
                if path.exists():
                    self.model_path = path
                    logger.info(f"Found model at {path}")
                    break
                    
            if not self.model_path:
                logger.warning(f"Could not find model file for {self.model_name}")
                
            # Create a temporary directory for audio files
            self.temp_dir = tempfile.mkdtemp(prefix="who_dey_tallk_")
            logger.info(f"Created temporary directory at {self.temp_dir}")
            
        except Exception as e:
            logger.error(f"Error initializing speech-to-text system: {e}")
    
    # --- New Stream Callback ---
    def _audio_callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            logger.warning(f"Sounddevice stream status: {status}")
        # Add incoming audio data to the queue as a copy
        self.audio_queue.put(indata.copy())

    # --- New Start/Stop Stream Methods ---
    def start_stream(self, device_id=None, channels=1, dtype='int16'):
        """Starts the non-blocking audio input stream."""
        if self.stream is not None and self.stream.active:
            logger.warning("Stream is already active.")
            return True
        
        try:
            logger.info(f"Starting audio stream (Device: {device_id or 'default'}, SR: {self.sample_rate}, Channels: {channels}, Dtype: {dtype})")
            self.stream_stop_event.clear() # Ensure event is clear before starting
            self.audio_queue = queue.Queue() # Clear queue on start

            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=channels,
                dtype=dtype,
                device=device_id,
                callback=self._audio_callback,
                # blocksize can be adjusted, 0 means variable block size based on callback timing
                # A fixed blocksize might be slightly simpler if needed, e.g., blocksize=int(self.sample_rate * 0.1) # 100ms blocks
            )
            self.stream.start()
            logger.info("Audio stream started.")
            return True
        except Exception as e:
            logger.error(f"Error starting audio stream: {e}", exc_info=True)
            self.stream = None
            return False

    def stop_stream(self):
        """Stops the audio input stream."""
        if self.stream is None or not self.stream.active:
            logger.info("Audio stream is not active.")
            return

        try:
            logger.info("Stopping audio stream...")
            self.stream_stop_event.set() # Signal any waiting reads to stop
            self.stream.stop()
            self.stream.close()
            logger.info("Audio stream stopped and closed.")
        except Exception as e:
            logger.error(f"Error stopping audio stream: {e}", exc_info=True)
        finally:
            self.stream = None
            # Clear the queue after stopping to prevent processing old data on restart
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            logger.debug("Audio queue cleared after stream stop.")

    # --- Modified record_audio_chunk ---
    def record_audio_chunk(self, duration=1.0, channels=1, dtype='int16'): # Removed device_id, handled by start_stream
        """
        Reads a chunk of audio data from the running stream's queue.

        Args:
            duration: Approximate duration in seconds to read.
            channels: Number of audio channels (should match stream).
            dtype: Data type (should match stream).

        Returns:
            numpy.ndarray: Audio data chunk or None if stream inactive or error.
        """
        if self.stream is None or not self.stream.active:
            logger.warning("Audio stream is not active. Cannot record chunk.")
            return None

        num_frames_needed = int(duration * self.sample_rate)
        collected_frames = 0
        audio_segments = []

        try:
            # Calculate a reasonable timeout based on duration
            # Add a buffer (e.g., 0.5s) to allow time for data arrival
            timeout_seconds = duration + 0.5 
            start_time = time.monotonic()

            while collected_frames < num_frames_needed:
                # Check if the stream stop event is set
                if self.stream_stop_event.is_set():
                    logger.info("Stream stop event detected during chunk recording.")
                    break # Exit loop if stop is requested

                # Calculate remaining timeout
                elapsed_time = time.monotonic() - start_time
                remaining_timeout = max(0.01, timeout_seconds - elapsed_time) # Ensure non-negative timeout

                try:
                    # Wait for data with timeout
                    segment = self.audio_queue.get(timeout=remaining_timeout)
                    audio_segments.append(segment)
                    collected_frames += segment.shape[0]
                except queue.Empty:
                    logger.warning(f"Audio queue empty after waiting {remaining_timeout:.2f}s. Returning collected data.")
                    break # Timeout reached, return what we have

            if not audio_segments:
                logger.debug("No audio segments collected for chunk.")
                return None

            # Concatenate segments and trim/pad to the exact duration
            full_audio = np.concatenate(audio_segments, axis=0)

            if full_audio.shape[0] >= num_frames_needed:
                # Trim if we collected too much
                final_chunk = full_audio[:num_frames_needed]
            else:
                # Pad with zeros if we collected too little (e.g., due to timeout or stop event)
                logger.warning(f"Collected fewer frames ({full_audio.shape[0]}) than needed ({num_frames_needed}). Padding with zeros.")
                padding_needed = num_frames_needed - full_audio.shape[0]
                # Ensure padding shape matches channels
                padding_shape = (padding_needed,) if channels == 1 else (padding_needed, channels)
                padding = np.zeros(padding_shape, dtype=dtype)
                final_chunk = np.concatenate((full_audio, padding), axis=0)
            
            logger.debug(f"Returning audio chunk with shape {final_chunk.shape}")
            return final_chunk

        except Exception as e:
            logger.error(f"Error reading audio chunk from queue: {e}", exc_info=True)
            return None

    def save_audio(self, audio_data, output_path=None):
        """
        Save audio data to a WAV file

        Args:
            audio_data: Audio data as numpy array
            output_path: Output file path (if None, a temporary file is created)

        Returns:
            str: Path to the saved audio file or None if failed
        """
        try:
            if audio_data is None:
                return None

            # Create a temporary file if output_path is not specified
            if output_path is None:
                if self.temp_dir is None:
                    self.temp_dir = tempfile.mkdtemp(prefix="who_dey_tallk_")
                    
                output_path = os.path.join(self.temp_dir, f"audio_{int(time.time())}.wav")
            
            # Save audio to WAV file
            # soundfile handles different dtypes automatically
            sf.write(output_path, audio_data, self.sample_rate)
            logger.debug(f"Audio data saved to {output_path}")

            return output_path

        except Exception as e:
            logger.error(f"Error saving audio: {e}", exc_info=True)
            return None
    
    def transcribe(self, audio_data_or_path):
        """
        Transcribe speech from audio

        Args:
            audio_data_or_path: Audio data as numpy array (int16 or float32) or path to audio file

        Returns:
            dict: Transcription result with text and metadata
                  or None if transcription failed
        """
        try:
            with self.lock:
                # Check if input is audio data or file path
                if isinstance(audio_data_or_path, np.ndarray):
                    # Save audio data to a temporary file
                    # save_audio handles int16 correctly
                    audio_path = self.save_audio(audio_data_or_path)
                    if audio_path is None:
                        logger.error("Failed to save audio data for transcription.")
                        return None
                elif isinstance(audio_data_or_path, (str, Path)):
                    # Use the provided file path
                    audio_path = str(audio_data_or_path)
                    if not os.path.exists(audio_path):
                        logger.error(f"Audio file not found: {audio_path}")
                        return None
                else:
                    logger.error("Invalid audio input type")
                    return None
                
                # Check if we have the Whisper CLI
                if self.whisper_cli and self.model_path:
                    # Use Whisper CLI for transcription
                    return self._transcribe_with_whisper_cli(audio_path)
                else:
                    # Try to use Python whisper if available
                    return self._transcribe_with_python_whisper(audio_path)
                    
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            return None
    
    def _transcribe_with_whisper_cli(self, audio_path):
        """
        Transcribe using Whisper CLI
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            dict: Transcription result
        """
        try:
            # Build command
            cmd = [
                str(self.whisper_cli),
                "-m", str(self.model_path),
                "-f", audio_path,
                "-l", self.language,
                "-t", "4",  # threads
                "--beam-size", "5",
                "-p", "1"   # processors
            ]
            
            # Run command
            logger.debug(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Whisper CLI error: {result.stderr}")
                return None
            
            # Parse output
            output = result.stdout.strip()
            
            # Extract the transcribed text
            text = ""
            for line in output.split("\n"):
                if "]" in line and "[" in line:
                    # Format is typically [timestamp] text
                    text_part = line.split("]", 1)[1].strip()
                    text += text_part + " "
            
            text = text.strip()
            
            return {
                "text": text,
                "language": self.language,
                "model": self.model_name
            }
            
        except Exception as e:
            logger.error(f"Error in Whisper CLI transcription: {e}")
            return None
    
    def _transcribe_with_python_whisper(self, audio_path):
        """
        Transcribe using Python Whisper library
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            dict: Transcription result
        """
        try:
            # Try to import whisper
            try:
                import whisper
            except ImportError:
                logger.error("Python whisper not installed and Whisper CLI not found")
                return None
            
            # Load model
            logger.info(f"Loading whisper model: {self.model_name}")
            model = whisper.load_model(self.model_name.replace("ggml-", ""))
            
            # Transcribe
            logger.info(f"Transcribing audio: {audio_path}")
            result = model.transcribe(
                audio_path, 
                language=self.language,
                fp16=False
            )
            
            return {
                "text": result.get("text", ""),
                "language": result.get("language", self.language),
                "model": self.model_name,
                "segments": result.get("segments", [])
            }
            
        except Exception as e:
            logger.error(f"Error in Python whisper transcription: {e}")
            return None

    # --- Add a cleanup method ---
    def close(self):
        """Clean up resources, especially the audio stream."""
        logger.info("Closing SpeechToText resources...")
        self.stop_stream()
        # Clean up temp dir if needed (consider if it should persist)
        # if self.temp_dir and os.path.exists(self.temp_dir):
        #     try:
        #         shutil.rmtree(self.temp_dir)
        #         logger.info(f"Removed temporary directory: {self.temp_dir}")
        #     except Exception as e:
        #         logger.error(f"Error removing temporary directory {self.temp_dir}: {e}")
        # self.temp_dir = None
    
    def __del__(self):
        """Clean up when object is destroyed"""
        self.close()
    
    @staticmethod
    def list_audio_devices():
        """List available audio devices"""
        try:
            devices = sd.query_devices()
            print("\nAvailable audio input devices:")
            print("-" * 60)
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    print(f"ID: {i}, Name: {device['name']}, "
                          f"Channels: {device['max_input_channels']}, "
                          f"Sample Rate: {device['default_samplerate']}")
                          
            return devices
            
        except Exception as e:
            logger.error(f"Error listing audio devices: {e}")
            return []