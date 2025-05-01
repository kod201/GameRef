#!/usr/bin/env python3
"""
Configuration Loader for Who Dey Tallk 2

Loads and validates configuration settings from YAML files
"""
import os
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Configuration loader class for Who Dey Tallk system"""
    
    def __init__(self, config_path):
        """
        Initialize the configuration loader
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = Path(config_path)
        self.config = None
    
    def load(self):
        """
        Load configuration from file
        
        Returns:
            dict: Configuration settings or default configuration if loading fails
        """
        try:
            if not self.config_path.exists():
                logger.warning(f"Configuration file not found: {self.config_path}")
                self.config = self._create_default_config()
                self._save_config()
                return self.config
            
            logger.info(f"Loading configuration from {self.config_path}")
            
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Validate configuration
            self._validate_config()
            
            return self.config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            
            # Return default configuration if loading fails
            self.config = self._create_default_config()
            return self.config
    
    def _validate_config(self):
        """Validate the configuration, filling in missing values with defaults"""
        if self.config is None:
            self.config = self._create_default_config()
            return
        
        # Ensure all required sections and settings are present
        default_config = self._create_default_config()
        
        for section, settings in default_config.items():
            if section not in self.config:
                logger.warning(f"Missing configuration section: {section}. Using defaults.")
                self.config[section] = settings
            elif isinstance(settings, dict):
                for key, value in settings.items():
                    if key not in self.config[section]:
                        logger.warning(f"Missing configuration setting: {section}.{key}. Using default: {value}")
                        self.config[section][key] = value
    
    def _save_config(self):
        """Save the current configuration to file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
                
            logger.info(f"Saved default configuration to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def _create_default_config(self):
        """
        Create a default configuration
        
        Returns:
            dict: Default configuration settings
        """
        return {
            "video": {
                "enabled": True,
                "device_id": 0,
                "width": 640,
                "height": 480,
                "fps": 30,
                "lip_sync_verification": True
            },
            "audio": {
                "device_id": None,  # Use default audio device
                "sample_rate": 16000,
                "chunk_duration": 3.0,
                "voice_biometrics_enabled": True,
                "transcription": {
                    "model": "tiny.en",
                    "language": "en"
                }
            },
            "face_recognition": {
                "model_path": "models/face_detection_model.bin",
                "known_faces_dir": "input/known_faces",
                "detection_threshold": 0.5,
                "recognition_threshold": 0.6
            },
            "lip_sync": {
                "model_path": "models/lip_sync_model.bin",
                "threshold": 0.7
            },
            "voice_biometrics": {
                "model_path": "models/voice_embedding_model.bin",
                "embeddings_path": "input/voice_embeddings",
                "threshold": 0.75
            },
            "speaker_matching": {
                "face_recognition_weight": 0.5,
                "lip_sync_weight": 0.3,
                "voice_biometrics_weight": 0.2,
                "min_confidence_threshold": 0.65,
                "unknown_speaker_threshold": 0.45
            },
            "database": {
                "path": "database/conversations.db",
                "max_history_days": 30
            },
            "monitoring": {
                "enabled": True,
                "refresh_rate": 1.0,
                "log_statistics": True,
                "stats_update_interval": 60
            },
            "output": {
                "save_transcripts": True,
                "transcript_path": "output/transcripts",
                "save_audio": False,
                "audio_path": "output/audio"
            }
        }