#!/usr/bin/env python3
"""
Configuration Management for Who Dey Tallk 2

Handles loading, saving and management of application configuration
"""
import os
import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigManager:
    """Configuration management for Who Dey Tallk system"""
    
    # Default configuration options
    DEFAULT_CONFIG = {
        "system": {
            "version": "2.0.0",
            "debug_mode": False,
            "log_level": "INFO",
            "data_dir": "data",
            "output_dir": "output"
        },
        "video": {
            "enabled": True,
            "device_id": 0,
            "width": 640,
            "height": 480,
            "fps": 30,
            "show_preview": True
        },
        "audio": {
            "enabled": True,
            "device_index": None,  # None means default device
            "sample_rate": 16000,
            "channels": 1,
            "chunk_size": 1024,
            "record_timeout": 0.5,
            "silence_threshold": 0.03,
            "silence_duration": 1.0
        },
        "whisper": {
            "model": "tiny.en",
            "model_path": "models/ggml-tiny.en.bin",
            "language": "en",
            "translate": False,
            "beam_size": 5,
            "use_gpu": True
        },
        "face_recognition": {
            "enabled": True,
            "model": "hog",  # hog or cnn
            "tolerance": 0.6,
            "num_jitters": 1,
            "known_faces_dir": "data/known_faces",
            "min_face_size": 20
        },
        "speaker_identification": {
            "enabled": True,
            "confidence_threshold": 0.65,
            "min_speaker_time": 1.0,
            "timeout_seconds": 5.0
        },
        "ui": {
            "theme": "dark",
            "font_size": 12,
            "show_metrics": True,
            "show_timestamps": True
        },
        "database": {
            "type": "sqlite",
            "path": "database/conversations.db",
            "max_history": 1000
        },
        "export": {
            "format": "json",
            "auto_export": False,
            "export_path": "output/conversations"
        }
    }
    
    def __init__(self, config_dir="config", config_file="config.json"):
        """
        Initialize configuration manager
        
        Args:
            config_dir: Directory for configuration files
            config_file: Name of the main configuration file
        """
        self.config_dir = config_dir
        self.config_file = config_file
        self.config_path = os.path.join(config_dir, config_file)
        
        # Ensure config directory exists
        os.makedirs(config_dir, exist_ok=True)
        
        # Load or create config
        self.config = self.load_config()
    
    def load_config(self):
        """
        Load configuration from file or create default
        
        Returns:
            dict: Configuration dictionary
        """
        # If config file exists, load it
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    logger.info(f"Configuration loaded from {self.config_path}")
                    
                    # Merge with defaults to ensure all keys exist
                    merged_config = self._merge_with_defaults(config)
                    return merged_config
                    
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
                logger.warning("Loading default configuration instead")
                return self._create_default_config()
        else:
            # Create and save default config
            logger.info("No configuration file found, creating default")
            return self._create_default_config()
    
    def _create_default_config(self):
        """
        Create default configuration
        
        Returns:
            dict: Default configuration dictionary
        """
        # Create a deep copy of default config
        config = json.loads(json.dumps(self.DEFAULT_CONFIG))
        
        # Save the default configuration
        self.save_config(config)
        
        return config
    
    def _merge_with_defaults(self, user_config):
        """
        Merge user configuration with defaults to ensure all settings exist
        
        Args:
            user_config: User configuration dictionary
            
        Returns:
            dict: Merged configuration
        """
        # Start with a deep copy of default config
        merged = json.loads(json.dumps(self.DEFAULT_CONFIG))
        
        # Update with user config values (recursive)
        def update_dict(target, source):
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    # Recursively update nested dictionaries
                    update_dict(target[key], value)
                else:
                    # Update or add the value
                    target[key] = value
        
        update_dict(merged, user_config)
        return merged
    
    def save_config(self, config=None):
        """
        Save configuration to file
        
        Args:
            config: Configuration to save (uses self.config if None)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if config is None:
            config = self.config
        
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Save config with pretty formatting
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)
                
            logger.info(f"Configuration saved to {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def get(self, path, default=None):
        """
        Get a configuration value by path
        
        Args:
            path: Dot-separated path to the value (e.g., "audio.device_index")
            default: Default value to return if path not found
            
        Returns:
            Value at path or default if not found
        """
        parts = path.split('.')
        value = self.config
        
        try:
            for part in parts:
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, path, value):
        """
        Set a configuration value by path
        
        Args:
            path: Dot-separated path to the value (e.g., "audio.device_index")
            value: Value to set
            
        Returns:
            bool: True if successful
        """
        parts = path.split('.')
        config = self.config
        
        # Navigate to the parent of the value to set
        for part in parts[:-1]:
            if part not in config:
                config[part] = {}
            config = config[part]
        
        # Set the value
        config[parts[-1]] = value
        
        # Save the updated config
        return self.save_config()
    
    def reset_to_defaults(self):
        """
        Reset configuration to defaults
        
        Returns:
            bool: True if successful
        """
        self.config = self._create_default_config()
        return self.save_config()
    
    def backup_config(self, backup_path=None):
        """
        Create a backup of the current configuration
        
        Args:
            backup_path: Path for the backup file, if None, uses config_path + .bak
            
        Returns:
            bool: True if successful
        """
        if not os.path.exists(self.config_path):
            logger.warning("No configuration file to backup")
            return False
        
        if backup_path is None:
            backup_path = self.config_path + ".bak"
        
        try:
            shutil.copy2(self.config_path, backup_path)
            logger.info(f"Configuration backed up to {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Error backing up configuration: {e}")
            return False
    
    def restore_backup(self, backup_path=None):
        """
        Restore configuration from backup
        
        Args:
            backup_path: Path to backup file, if None, uses config_path + .bak
            
        Returns:
            bool: True if successful
        """
        if backup_path is None:
            backup_path = self.config_path + ".bak"
        
        if not os.path.exists(backup_path):
            logger.error(f"Backup file not found: {backup_path}")
            return False
        
        try:
            shutil.copy2(backup_path, self.config_path)
            self.config = self.load_config()
            logger.info(f"Configuration restored from {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Error restoring configuration: {e}")
            return False
    
    def get_all(self):
        """
        Get the entire configuration
        
        Returns:
            dict: The entire configuration dictionary
        """
        return self.config
    
    def create_directories(self):
        """
        Create all directories specified in the configuration
        
        Returns:
            bool: True if all directories were created successfully
        """
        try:
            # Create data directory
            data_dir = self.get("system.data_dir", "data")
            os.makedirs(data_dir, exist_ok=True)
            
            # Create output directory
            output_dir = self.get("system.output_dir", "output")
            os.makedirs(output_dir, exist_ok=True)
            
            # Create known faces directory
            faces_dir = self.get("face_recognition.known_faces_dir", "data/known_faces")
            os.makedirs(faces_dir, exist_ok=True)
            
            # Create database directory
            db_path = self.get("database.path", "database/conversations.db")
            db_dir = os.path.dirname(db_path)
            os.makedirs(db_dir, exist_ok=True)
            
            # Create exports directory
            export_path = self.get("export.export_path", "output/conversations")
            os.makedirs(export_path, exist_ok=True)
            
            logger.info("All configuration directories created")
            return True
            
        except Exception as e:
            logger.error(f"Error creating directories: {e}")
            return False