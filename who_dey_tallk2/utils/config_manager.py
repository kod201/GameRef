#!/usr/bin/env python3
"""
Configuration Manager for Who Dey Tallk v2

Handles loading, parsing and accessing configuration settings from YAML files
and command line overrides.
"""

import os
import yaml
import logging
from pathlib import Path

logger = logging.getLogger('config')

class ConfigManager:
    """Manages application configuration from YAML files and command line arguments"""
    
    def __init__(self, config_file=None):
        """
        Initialize configuration manager
        
        Args:
            config_file: Path to YAML configuration file (default: config/default.yaml)
        """
        self.config = {}
        self.base_dir = Path(__file__).parent.parent.resolve()
        
        # Set default config file if none provided
        if not config_file:
            config_file = self.base_dir / "config" / "default.yaml"
        elif not os.path.isabs(config_file):
            config_file = self.base_dir / config_file
        
        # Load configuration
        self.load_config(config_file)
    
    def load_config(self, config_file):
        """
        Load configuration from a YAML file
        
        Args:
            config_file: Path to YAML configuration file
        """
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    self.config = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration from {config_file}")
            else:
                logger.warning(f"Config file not found: {config_file}")
                self.config = {}
                
            # Set some defaults if not in config
            if "paths" not in self.config:
                self.config["paths"] = {}
                
            if "base_dir" not in self.config["paths"]:
                self.config["paths"]["base_dir"] = str(self.base_dir)
                
            if "output_dir" not in self.config["paths"]:
                self.config["paths"]["output_dir"] = str(self.base_dir / "output")
                
            if "models_dir" not in self.config["paths"]:
                self.config["paths"]["models_dir"] = str(self.base_dir / "models")
                
        except Exception as e:
            logger.error(f"Error loading config file {config_file}: {e}")
            self.config = {}
    
    def get(self, key_path, default=None):
        """
        Get a configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to the config value (e.g., "database.host")
            default: Default value if key not found
            
        Returns:
            The configuration value or default if not found
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path, value):
        """
        Set a configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to the config value (e.g., "database.host")
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to the correct location
        for i, key in enumerate(keys[:-1]):
            if key not in config:
                config[key] = {}
            config = config[key]
            
        # Set the value
        config[keys[-1]] = value
    
    def update_from_args(self, args):
        """
        Update configuration from command line arguments
        
        Args:
            args: Parsed command line arguments
        """
        # Convert args to dictionary
        args_dict = vars(args)
        
        # Map command line args to config paths
        arg_mappings = {
            "video_device": "input.video.device_id",
            "audio_device": "input.audio.device_id",
            "face_model": "models.face_recognition.model",
            "speech_model": "models.speech_recognition.model",
            "voice_model": "models.voice_biometrics.model",
            "output_dir": "paths.output_dir",
            "save_video": "output.save_video",
            "save_audio": "output.save_audio",
            "debug": "debug"
        }
        
        # Update config from args
        for arg, config_path in arg_mappings.items():
            if arg in args_dict and args_dict[arg] is not None:
                self.set(config_path, args_dict[arg])
        
        # Special handling for enabling/disabling features
        if args_dict.get("no_video"):
            self.set("features.video_processing.enabled", False)
            
        if args_dict.get("no_audio"):
            self.set("features.audio_processing.enabled", False)
            
        if args_dict.get("no_database"):
            self.set("features.database_storage.enabled", False)