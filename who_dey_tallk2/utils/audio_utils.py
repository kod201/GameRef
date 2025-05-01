#!/usr/bin/env python3
"""
Audio Utilities for Who Dey Tallk 2

Provides helper functions for audio processing, analysis, and visualization
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import librosa
import librosa.display
import soundfile as sf
import logging

logger = logging.getLogger(__name__)

def calculate_audio_level(audio_chunk, percentile=95):
    """
    Calculate the audio level from an audio chunk
    
    Args:
        audio_chunk: Audio data as numpy array
        percentile: Percentile to use for level calculation (avoid outliers)
        
    Returns:
        float: Audio level from 0.0 to 1.0
    """
    if audio_chunk is None or len(audio_chunk) == 0:
        return 0.0
    
    # Calculate absolute amplitude
    abs_data = np.abs(audio_chunk)
    
    # Use percentile to avoid being skewed by outliers
    level = np.percentile(abs_data, percentile)
    
    # Normalize to 0-1 range
    normalized_level = min(1.0, level * 5.0)  # Scale for better sensitivity
    
    return normalized_level

def detect_silence(audio_chunk, threshold=0.01, min_duration_samples=1000):
    """
    Detect silent segments in audio
    
    Args:
        audio_chunk: Audio data as numpy array
        threshold: Amplitude threshold below which is considered silence
        min_duration_samples: Minimum duration of silence in samples
        
    Returns:
        list: List of (start_sample, end_sample) tuples for silent segments
    """
    if audio_chunk is None or len(audio_chunk) == 0:
        return []
    
    # Calculate absolute amplitude
    abs_data = np.abs(audio_chunk)
    
    # Find silent segments
    is_silent = abs_data < threshold
    
    # Find transitions
    transitions = np.where(np.diff(is_silent.astype(int)) != 0)[0] + 1
    
    if len(transitions) == 0:
        # Either all silent or all non-silent
        if is_silent[0]:
            return [(0, len(audio_chunk))]
        else:
            return []
    
    # Add start and end points
    if is_silent[0]:
        transitions = np.concatenate(([0], transitions))
    if is_silent[-1]:
        transitions = np.concatenate((transitions, [len(audio_chunk)]))
    
    # Group into pairs
    silent_segments = []
    for i in range(0, len(transitions), 2):
        if i + 1 < len(transitions):
            start, end = transitions[i], transitions[i+1]
            if end - start >= min_duration_samples:
                silent_segments.append((start, end))
    
    return silent_segments

def split_audio_on_silence(audio_chunk, sample_rate, threshold=0.01, min_silence_duration=0.5):
    """
    Split audio into segments based on silence
    
    Args:
        audio_chunk: Audio data as numpy array
        sample_rate: Sample rate of audio data
        threshold: Amplitude threshold for silence detection
        min_silence_duration: Minimum silence duration in seconds
        
    Returns:
        list: List of audio segments as numpy arrays
    """
    min_silence_samples = int(min_silence_duration * sample_rate)
    
    # Detect silent segments
    silent_segments = detect_silence(audio_chunk, threshold, min_silence_samples)
    
    if not silent_segments:
        # No silence detected, return the entire chunk
        return [audio_chunk]
    
    # Split audio at silent segments
    audio_segments = []
    last_end = 0
    
    for start, end in silent_segments:
        # Add non-silent segment before this silent segment
        if start > last_end:
            audio_segments.append(audio_chunk[last_end:start])
        last_end = end
    
    # Add final segment if there is one
    if last_end < len(audio_chunk):
        audio_segments.append(audio_chunk[last_end:])
    
    # Remove empty segments
    audio_segments = [seg for seg in audio_segments if len(seg) > 0]
    
    return audio_segments

def visualize_audio(audio_chunk, sample_rate, title="Audio Waveform"):
    """
    Create a visualization of audio data
    
    Args:
        audio_chunk: Audio data as numpy array
        sample_rate: Sample rate of audio data
        title: Title for the plot
        
    Returns:
        matplotlib.figure.Figure: Figure object or None if visualization failed
    """
    try:
        plt.figure(figsize=(10, 4))
        
        # Plot waveform
        librosa.display.waveshow(audio_chunk, sr=sample_rate)
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        
        return plt.gcf()
        
    except Exception as e:
        logger.error(f"Error visualizing audio: {e}")
        return None

def save_audio_visualization(audio_chunk, sample_rate, output_path, title="Audio Waveform"):
    """
    Save audio visualization to a file
    
    Args:
        audio_chunk: Audio data as numpy array
        sample_rate: Sample rate of audio data
        output_path: Path to save the visualization
        title: Title for the plot
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        fig = visualize_audio(audio_chunk, sample_rate, title)
        
        if fig:
            fig.savefig(output_path)
            plt.close(fig)
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error saving audio visualization: {e}")
        return False

def compute_mfccs(audio_chunk, sample_rate, n_mfcc=20):
    """
    Compute MFCCs (Mel Frequency Cepstral Coefficients) from audio
    
    Args:
        audio_chunk: Audio data as numpy array
        sample_rate: Sample rate of audio data
        n_mfcc: Number of MFCCs to extract
        
    Returns:
        numpy.ndarray: MFCC features
    """
    try:
        # Compute MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio_chunk, 
            sr=sample_rate,
            n_mfcc=n_mfcc
        )
        
        return mfccs
        
    except Exception as e:
        logger.error(f"Error computing MFCCs: {e}")
        return None

def detect_speech_segments(audio_chunk, sample_rate, threshold=0.01, min_duration=0.3):
    """
    Detect segments containing speech in audio
    
    Args:
        audio_chunk: Audio data as numpy array
        sample_rate: Sample rate of audio data
        threshold: Energy threshold for speech detection
        min_duration: Minimum speech duration in seconds
        
    Returns:
        list: List of (start_time, end_time) tuples in seconds
    """
    try:
        # Convert threshold to amplitude
        amplitude_threshold = threshold
        
        # Calculate energy
        energy = np.abs(audio_chunk)
        
        # Find speech segments
        is_speech = energy > amplitude_threshold
        
        # Find transitions
        transitions = np.where(np.diff(is_speech.astype(int)) != 0)[0] + 1
        
        if len(transitions) == 0:
            # Either all speech or all silence
            if is_speech[0]:
                return [(0, len(audio_chunk) / sample_rate)]
            else:
                return []
        
        # Add start and end points
        if is_speech[0]:
            transitions = np.concatenate(([0], transitions))
        if is_speech[-1]:
            transitions = np.concatenate((transitions, [len(audio_chunk)]))
        
        # Group into pairs
        min_samples = int(min_duration * sample_rate)
        speech_segments = []
        
        for i in range(0, len(transitions), 2):
            if i + 1 < len(transitions):
                start, end = transitions[i], transitions[i+1]
                if end - start >= min_samples:
                    # Convert to seconds
                    start_time = start / sample_rate
                    end_time = end / sample_rate
                    speech_segments.append((start_time, end_time))
        
        return speech_segments
        
    except Exception as e:
        logger.error(f"Error detecting speech segments: {e}")
        return []

def adjust_audio_speed(audio_chunk, sample_rate, speed_factor):
    """
    Adjust the speed of audio
    
    Args:
        audio_chunk: Audio data as numpy array
        sample_rate: Sample rate of audio data
        speed_factor: Factor to adjust speed (1.0 = normal, 0.5 = half speed, 2.0 = double speed)
        
    Returns:
        tuple: (adjusted_audio, new_sample_rate)
    """
    try:
        if speed_factor == 1.0:
            return audio_chunk, sample_rate
            
        # Time-stretch the audio
        stretched = librosa.effects.time_stretch(audio_chunk, rate=speed_factor)
        
        return stretched, sample_rate
        
    except Exception as e:
        logger.error(f"Error adjusting audio speed: {e}")
        return audio_chunk, sample_rate

def extract_audio_features(audio_chunk, sample_rate):
    """
    Extract various audio features for analysis
    
    Args:
        audio_chunk: Audio data as numpy array
        sample_rate: Sample rate of audio data
        
    Returns:
        dict: Dictionary of audio features
    """
    try:
        features = {}
        
        # Basic statistics
        features['rms'] = np.sqrt(np.mean(np.square(audio_chunk)))
        features['peak'] = np.max(np.abs(audio_chunk))
        features['duration'] = len(audio_chunk) / sample_rate
        
        # Spectral features
        if len(audio_chunk) > 0:
            # Spectral centroid
            cent = librosa.feature.spectral_centroid(y=audio_chunk, sr=sample_rate)
            features['spectral_centroid'] = np.mean(cent)
            
            # Spectral bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(y=audio_chunk, sr=sample_rate)
            features['spectral_bandwidth'] = np.mean(bandwidth)
            
            # Zero crossing rate - useful for voice/music discrimination
            zcr = librosa.feature.zero_crossing_rate(audio_chunk)
            features['zero_crossing_rate'] = np.mean(zcr)
            
            # MFCC
            mfccs = compute_mfccs(audio_chunk, sample_rate)
            if mfccs is not None:
                features['mfcc_mean'] = np.mean(mfccs, axis=1)
                features['mfcc_var'] = np.var(mfccs, axis=1)
        
        return features
        
    except Exception as e:
        logger.error(f"Error extracting audio features: {e}")
        return {}