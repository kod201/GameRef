"""
Integration with Whisper.cpp for speech recognition.
"""
import os
import subprocess
import platform
import shutil
import json
from pathlib import Path
from typing import Dict, Optional, Union, List, Tuple

class WhisperManager:
    """Manages Whisper.cpp integration and transcription."""
    
    def __init__(
        self,
        model_name: str = "tiny.en",
        build_dir: str = None,
        models_dir: str = None,
        whisper_cpp_repo: str = "https://github.com/ggerganov/whisper.cpp.git"
    ):
        """
        Initialize Whisper manager.
        
        Args:
            model_name: Name of the Whisper model to use (e.g., "tiny.en")
            build_dir: Directory to build/store Whisper.cpp
            models_dir: Directory to store model files
            whisper_cpp_repo: Repository URL for Whisper.cpp
        """
        self.model_name = model_name
        
        # Set up directories
        base_dir = Path(__file__).parent.absolute()
        self.build_dir = Path(build_dir) if build_dir else base_dir / "build" / "whisper.cpp"
        self.models_dir = Path(models_dir) if models_dir else base_dir / "models"
        self.whisper_cpp_repo = whisper_cpp_repo
        
        # Ensure directories exist
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths for executables and models
        # Updated path to use the whisper-cli executable instead of main (which is deprecated)
        self.whisper_executable = self.build_dir / "build" / "bin" / "whisper-cli"
        if platform.system() == "Windows":
            self.whisper_executable = self.build_dir / "build" / "bin" / "whisper-cli.exe"
            
        self.model_path = self.models_dir / f"ggml-{model_name}.bin"
        self.model_download_url = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{model_name}.bin"
    
    def is_whisper_built(self) -> bool:
        """Check if Whisper.cpp is already built."""
        return self.whisper_executable.exists()
    
    def is_model_downloaded(self) -> bool:
        """Check if the specified model is already downloaded."""
        return self.model_path.exists()
    
    def build_whisper_cpp(self) -> bool:
        """
        Build Whisper.cpp from source.
        
        Returns:
            True if build successful, False otherwise
        """
        try:
            current_dir = os.getcwd()
            
            # Clone the repository if not exists
            if not (self.build_dir / ".git").exists():
                print(f"Cloning Whisper.cpp repository to {self.build_dir}...")
                os.makedirs(self.build_dir.parent, exist_ok=True)
                subprocess.run(
                    ["git", "clone", self.whisper_cpp_repo, str(self.build_dir)],
                    check=True
                )
            
            # Build Whisper.cpp
            os.chdir(str(self.build_dir))
            print("Building Whisper.cpp...")
            
            # Use make on Unix-like systems, cmake on Windows
            if platform.system() in ["Linux", "Darwin"]:
                result = subprocess.run(["make"], check=True)
            else:
                # For Windows, use CMake
                os.makedirs("build", exist_ok=True)
                os.chdir("build")
                subprocess.run(["cmake", ".."], check=True)
                subprocess.run(["cmake", "--build", ".", "--config", "Release"], check=True)
                
                # Copy executable to expected location
                if os.path.exists("Release/main.exe"):
                    shutil.copy("Release/main.exe", "../main.exe")
            
            os.chdir(current_dir)
            return self.is_whisper_built()
        
        except Exception as e:
            print(f"Error building Whisper.cpp: {e}")
            os.chdir(current_dir)
            return False
    
    def download_model(self) -> bool:
        """
        Download the specified Whisper model.
        
        Returns:
            True if download successful, False otherwise
        """
        if self.is_model_downloaded():
            print(f"Model already downloaded at {self.model_path}")
            return True
            
        print(f"Downloading {self.model_name} model...")
        try:
            # Try using curl (preinstalled on macOS, many Linux distros)
            subprocess.run([
                "curl", "-L", self.model_download_url,
                "-o", str(self.model_path)
            ], check=True)
            
            return self.is_model_downloaded()
        except Exception as e:
            print(f"Error downloading model with curl: {e}")
            
            try:
                # Try with Python's urllib as a fallback
                import urllib.request
                print("Trying download with urllib...")
                urllib.request.urlretrieve(self.model_download_url, str(self.model_path))
                return self.is_model_downloaded()
            except Exception as e2:
                print(f"Error downloading model with urllib: {e2}")
                return False
    
    def ensure_ready(self) -> bool:
        """
        Ensure Whisper.cpp is built and model is downloaded.
        
        Returns:
            True if everything is ready, False otherwise
        """
        if not self.is_whisper_built():
            if not self.build_whisper_cpp():
                return False
        
        if not self.is_model_downloaded():
            if not self.download_model():
                return False
        
        return True
    
    def transcribe_audio(self, audio_file: str, options: Optional[Dict] = None) -> Dict:
        """
        Transcribe speech in an audio file.
        
        Args:
            audio_file: Path to the audio file (WAV)
            options: Additional transcription options
            
        Returns:
            Dictionary with transcription results and metadata
        """
        if not self.ensure_ready():
            return {
                "error": "Whisper.cpp or model not ready. Please check setup.",
                "transcript": "",
                "success": False
            }
        
        # Default options
        opts = {
            "language": "en",  # Language code
            "translate": False,  # Whether to translate to English
            "max_len": 0,       # Maximum segment length (0 = unlimited)
            "max_tokens": 32,   # Maximum tokens per chunk
            "beam_size": 5,     # Beam size for beam search
            "threads": os.cpu_count() or 4  # Number of threads to use
        }
        
        # Update with user options if provided
        if options:
            opts.update(options)
            
        try:
            # Prepare command
            cmd = [
                str(self.whisper_executable),
                "-m", str(self.model_path),
                "-f", str(audio_file),
                "-l", opts["language"],
                "-t", str(opts["threads"]),
                "--max-len", str(opts["max_len"]),
                "--max-tokens", str(opts["max_tokens"]),
                "--beam-size", str(opts["beam_size"]),
                "-ojson"  # Output in JSON format
            ]
            
            if opts["translate"]:
                cmd.append("--translate")
            
            # Run transcription
            print(f"Running transcription with {self.model_name} model...")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse JSON output if possible
            try:
                output_data = json.loads(result.stdout)
                output_data["success"] = True
                return output_data
            except json.JSONDecodeError:
                # Fall back to raw output if not valid JSON
                return {
                    "transcript": result.stdout.strip(),
                    "success": True
                }
                
        except subprocess.CalledProcessError as e:
            error_msg = f"Transcription failed: {str(e)}\n{e.stderr}"
            print(error_msg)
            return {
                "error": error_msg,
                "transcript": "",
                "success": False
            }
        except Exception as e:
            error_msg = f"Error during transcription: {str(e)}"
            print(error_msg)
            return {
                "error": error_msg,
                "transcript": "",
                "success": False
            }