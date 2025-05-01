# Who Dey Tallk 2

A comprehensive speaker identification and conversation tracking system that combines video processing with audio analysis to determine who is speaking.

## Overview

Who Dey Tallk 2 is a multi-modal system that:

1. Captures video and audio from a meeting or conversation
2. Identifies faces in the video feed
3. Detects lip movement to determine who is speaking
4. Uses voice biometrics to identify speakers through their voice alone
5. Transcribes spoken content using speech-to-text
6. Associates transcribed text with the correct speaker
7. Stores conversations with speaker attributions in a database

The system can work in both audio+video mode and audio-only mode, making it versatile for different environments.

## Architecture

The system is composed of several interconnected components:

- **Video Capture**: Captures frames from the camera
- **Face Recognition**: Identifies known faces in video frames
- **Lip Sync Detection**: Detects mouth movement to determine who is speaking
- **Voice Biometrics**: Identifies speakers through voice characteristics
- **Speech-to-Text**: Transcribes speech to text using Whisper
- **Speaker Matching**: Decision engine that combines all identification methods
- **Database**: Stores conversations and speaker profiles
- **Monitoring**: Provides real-time system statistics

## Requirements

- Python 3.8 or higher
- OpenCV for video processing
- PyTorch (optional, for more advanced face/voice models)
- Whisper.cpp or OpenAI Whisper for speech-to-text
- SQLite for database storage
- Additional libraries: librosa, sounddevice, psutil, numpy, etc.

See `requirements.txt` for a complete list of dependencies.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/who_dey_tallk2.git
cd who_dey_tallk2
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Install Whisper.cpp (for faster transcription):

```bash
# Clone and build whisper.cpp
mkdir -p build
cd build
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
make
```

5. Download the Whisper model:

```bash
mkdir -p models
# For tiny.en model
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin -O models/ggml-tiny.en.bin
```

6. Configure the system by editing `config/config.yaml`

## Usage

### Starting the system

```bash
python main.py
```

### Configuration options

```bash
# List available audio and video devices
python main.py --list-devices

# Use a specific configuration file
python main.py --config path/to/config.yaml

# Add a new person to face database
python main.py --add-face "person_id,Person Name,path/to/image.jpg"

# Add a new voice to voice database
python main.py --add-voice "person_id,Person Name,path/to/voice_sample.wav"

# Run in debug mode for more verbose logging
python main.py --debug
```

## Customization

### Adding known faces

1. Create a directory for the person in `input/known_faces/person_id/`
2. Add a `name.txt` file containing their name
3. Add one or more `.jpg` images of their face

### Adding known voices

1. Create a directory for the person in `input/voice_embeddings/person_id/`
2. Add a `name.txt` file containing their name
3. Add one or more `.wav` audio samples of their voice

### Configuration options

Edit `config/config.yaml` to customize:

- Audio input device and parameters
- Video input device and parameters
- Face recognition sensitivity
- Lip sync detection parameters
- Voice biometrics settings
- Speaker matching weights
- Database configuration
- Monitoring settings

## Key Features

- **Multi-modal identification**: Combines video and audio evidence to determine who is speaking
- **Persistent speaker profiles**: Remembers faces and voices of known speakers
- **Real-time processing**: Processes video and audio in real-time
- **Lip-sync verification**: Verifies that the identified face is actually speaking
- **Voice biometrics**: Can identify speakers by voice alone when faces are not visible
- **SQL database storage**: Stores all conversations with speaker attributions
- **Real-time monitoring**: Visual dashboard showing system performance

## Project Structure

```
who_dey_tallk2/
├── build/               # Build files for dependencies like whisper.cpp
├── config/              # Configuration files
├── database/            # Database files and management
├── input/               # Input files (known faces, voice samples)
│   ├── known_faces/
│   └── voice_embeddings/
├── logic/               # Core logic components
│   ├── face_recognition.py
│   ├── lip_sync_detector.py
│   ├── speaker_matching.py
│   ├── speech_to_text.py
│   ├── video_capture.py
│   └── voice_biometrics.py
├── models/              # Model files for various components
├── output/              # Output files (transcripts, recordings)
├── utils/               # Utility modules
│   ├── config_loader.py
│   └── monitoring.py
└── main.py              # Main entry point
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the original "who_dey_tallk" project
- Uses OpenAI's Whisper model for speech recognition
- Built with various open-source computer vision and audio processing libraries
