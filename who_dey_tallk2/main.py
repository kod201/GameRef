#!/usr/bin/env python3
"""
WHO DEY TALLK 2 - Conversation Detection and Speaker Identification System

This system detects, identifies, and records conversations from live video and audio feeds.
It tags each speaker with an ID and name, and associates spoken conversation with the
correct person in persistent memory (SQL database).

The system can work with:
- Video + Audio: Uses face recognition with lip-sync verification
- Audio only: Uses voice biometrics for speaker identification

Author: Stephen Ohakanu
Date: April 28, 2025
"""

import os
import sys
import time
import logging
import threading
import argparse
import numpy as np
import cv2
import webrtcvad
import uuid
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('who_dey_tallk')

# Ensure our modules can be imported
current_dir = Path(__file__).parent.resolve()
sys.path.append(str(current_dir))

# Import project modules
from logic.speech_to_text import SpeechToText
from logic.face_recognition import FaceRecognizer
from logic.lip_sync_detector import LipSyncDetector
from logic.voice_biometrics import VoiceBiometricsEngine
from logic.speaker_matching import SpeakerMatcher
from logic.video_capture import VideoCapture
from utils.config_loader import ConfigLoader

# Correct import for the modified DatabaseManager
from database.database_manager import DatabaseManager


class WhoDeyTallk:
    """
    Main application class for Who Dey Tallk system.
    Coordinates all components for conversation detection and speaker identification.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the Who Dey Tallk system
        
        Args:
            config_path: Path to configuration file (default: config/config.yaml)
        """
        self.base_dir = Path(__file__).parent.resolve()
        
        # Load configuration
        if config_path is None:
            config_path = self.base_dir / "config" / "config.yaml"
        
        self.config = ConfigLoader(config_path).load()

        # Initialize database first, passing the loaded config dictionary
        self.db = DatabaseManager(config=self.config)

        # Set up component flags based on configuration
        self.use_video = self.config.get("video", {}).get("enabled", True)
        self.use_voice_biometrics = self.config.get("audio", {}).get("voice_biometrics_enabled", True)
        self.use_lip_sync = self.config.get("video", {}).get("lip_sync_verification", True)
        self.show_video_window = self.config.get("monitoring", {}).get("show_video_window", True)

        # --- Add Voice Enroll/Enrich Config Flags ---
        self.voice_enroll_known_face_enabled = self.config.get("voice_biometrics", {}).get("enroll_if_known_face", True)
        self.voice_enrichment_enabled = self.config.get("voice_biometrics", {}).get("enrich_if_voice_match", True)
        self.voice_enroll_enrich_interval = self.config.get("voice_biometrics", {}).get("enroll_enrich_interval_sec", 10.0) # Interval in seconds
        # --- End Add ---

        # VAD Configuration
        self.vad_aggressiveness = self.config.get("audio", {}).get("vad_aggressiveness", 3) # Default to max aggressiveness
        self.vad_frame_duration_ms = self.config.get("audio", {}).get("vad_frame_duration_ms", 30) # VAD frame duration (10, 20, or 30)
        self.vad_sample_rate = self.config.get("audio", {}).get("sample_rate", 16000) # Ensure VAD uses correct sample rate
        self.vad_bytes_per_sample = 2 # Assuming 16-bit PCM audio

        # Validate VAD frame duration
        if self.vad_frame_duration_ms not in [10, 20, 30]:
            logger.warning(f"Invalid VAD frame duration ({self.vad_frame_duration_ms}ms). Defaulting to 30ms.")
            self.vad_frame_duration_ms = 30
        self.vad_frame_size = int(self.vad_sample_rate * (self.vad_frame_duration_ms / 1000.0))
        self.vad_frame_bytes = self.vad_frame_size * self.vad_bytes_per_sample


        # Initialize components
        self._init_components()

        # Internal state
        self.running = False
        self.threads = []
        # Remove self.monitor initialization here if replacing with OpenCV window
        # self.monitor = None

        # Shared state for OpenCV display (protected by lock)
        self.display_lock = threading.Lock()
        self.latest_frame = None
        self.latest_face_results = {}
        self.latest_lip_sync_scores = {}
        self.latest_speaker_result = {}
        self.latest_transcription = ""

        # --- Remove Enrollment Cooldown Attributes ---
        # Remove: self.last_enrollment_attempt_time = 0.0
        # Remove: self.enrollment_cooldown = 5.0
        # --- End Remove ---

        # --- Add Automatic Enroll/Enrich Attributes ---
        self.last_enroll_enrich_attempt_times = {} # Tracks last attempt time per face ID
        self.enroll_enrich_interval = self.config.get("face_recognition", {}).get("enroll_enrich_interval_sec", 15.0) # Interval in seconds
        # --- End Add ---

        # --- Add Voice Enroll/Enrich Cooldown Tracking ---
        self.last_voice_enroll_enrich_attempt_times = {} # Tracks last attempt time per speaker ID for voice
        # --- End Add ---


    def _init_components(self):
        """Initialize all system components based on configuration"""
        logger.info("Initializing Who Dey Tallk components...")

        # VAD initialization...
        try:
            self.vad = webrtcvad.Vad(self.vad_aggressiveness)
            logger.info(f"VAD initialized with aggressiveness {self.vad_aggressiveness} and {self.vad_frame_duration_ms}ms frames.")
        except Exception as e:
            logger.error(f"Failed to initialize VAD: {e}. VAD will be disabled.", exc_info=True)
            self.vad = None

        # Initialize speech-to-text
        self.stt = SpeechToText(
            model_name=self.config["audio"]["transcription"]["model"],
            language=self.config["audio"]["transcription"]["language"],
            sample_rate=self.config["audio"]["sample_rate"]
        )
        # Store audio config needed for starting stream
        self.audio_device_id = self.config.get("audio", {}).get("device_id")
        self.audio_channels = 1 # Assuming mono for now, adjust if config allows stereo
        self.audio_dtype = 'int16' # Assuming int16 based on previous code

        # Initialize video processing if enabled
        if self.use_video:
            self.video_capture = VideoCapture(
                device_id=self.config["video"]["device_id"],
                width=self.config["video"]["width"],
                height=self.config["video"]["height"],
                fps=self.config["video"]["fps"]
            )
            
            # Face recognition - Use model_path directly from config
            face_rec_config = self.config["face_recognition"]
            model_path_str = face_rec_config.get("model_path", "models/yolov8n.pt") # Default added
            model_full_path = self.base_dir / model_path_str

            if not model_full_path.exists():
                logger.warning(f"Face recognition model not found at {model_full_path}. FaceRecognizer might fail or use fallback.")
                # Pass the potentially non-existent path, let FaceRecognizer handle it
                model_path_arg = str(model_full_path)
            else:
                logger.info(f"Using face recognition model from {model_full_path}")
                model_path_arg = str(model_full_path)

            self.face_recognizer = FaceRecognizer(
                model_path=model_path_arg, # Pass the specific model file path
                known_faces_dir=str(self.base_dir / face_rec_config["known_faces_dir"]),
                detection_threshold=face_rec_config["detection_threshold"],
                recognition_threshold=face_rec_config["recognition_threshold"]
            )

            # Lip sync detection if enabled
            if self.use_lip_sync:
                 # Assuming lip sync model is also in the main models dir or specified path
                 lip_sync_config = self.config["lip_sync"]
                 lip_sync_model_str = lip_sync_config.get("model_path", "models/lip_sync_model.pth") # Default added
                 lip_sync_model_path = self.base_dir / lip_sync_model_str
                 if not lip_sync_model_path.exists():
                      logger.warning(f"Lip sync model not found at {lip_sync_model_path}. Lip sync will be disabled or fail.")
                      # Handle missing model - maybe disable lip sync?
                      self.lip_sync_detector = None
                 else:
                     self.lip_sync_detector = LipSyncDetector(
                        model_path=str(lip_sync_model_path),
                        threshold=lip_sync_config["threshold"]
                     )
            else:
                self.lip_sync_detector = None
        else:
            self.video_capture = None
            self.face_recognizer = None
            self.lip_sync_detector = None
        
        # Initialize voice biometrics if enabled
        if self.use_voice_biometrics:
             # Use model_path directly from config
             voice_bio_config = self.config["voice_biometrics"]
             voice_model_str = voice_bio_config.get("model_path", "models/voice_embedding_model.pt") # Default added
             voice_model_path = self.base_dir / voice_model_str
             if not voice_model_path.exists():
                  logger.warning(f"Voice biometrics model not found at {voice_model_path}. Voice biometrics might fail.")

             self.voice_biometrics = VoiceBiometricsEngine(
                # Remove model_path as it's loaded internally by SpeechBrain
                # model_path=str(voice_model_path), 
                embeddings_path=str(self.base_dir / voice_bio_config["embeddings_path"]),
                threshold=voice_bio_config["threshold"]
             )
        else:
            self.voice_biometrics = None
        
        # Initialize speaker matcher (decision engine)
        self.speaker_matcher = SpeakerMatcher(
            face_recognition_weight=self.config["speaker_matching"]["face_recognition_weight"],
            lip_sync_weight=self.config["speaker_matching"]["lip_sync_weight"],
            voice_biometrics_weight=self.config["speaker_matching"]["voice_biometrics_weight"]
        )

        # Remove monitor initialization here if replacing
        # self.monitor = None

        logger.info("All components initialized successfully")
    
    def _is_speech(self, audio_chunk_np):
        """Check if the audio chunk contains speech using VAD."""
        if not self.vad or audio_chunk_np is None or audio_chunk_np.size == 0:
            logger.debug("VAD not available or audio chunk is empty, skipping VAD check.")
            return True # Assume speech if VAD is disabled or chunk is invalid for VAD

        try:
            # Ensure audio is mono and 16-bit PCM for VAD
            if audio_chunk_np.ndim > 1:
                 audio_chunk_np = np.mean(audio_chunk_np, axis=1) # Convert stereo to mono if needed

            # Ensure correct dtype (int16 for VAD)
            if audio_chunk_np.dtype != np.int16:
                 # Scale float audio (-1.0 to 1.0) to int16 (-32768 to 32767)
                 if np.issubdtype(audio_chunk_np.dtype, np.floating):
                      audio_chunk_np = (audio_chunk_np * 32767).astype(np.int16)
                 else:
                      # Attempt conversion, might raise error for incompatible types
                      logger.warning(f"Audio chunk is not float or int16 ({audio_chunk_np.dtype}), attempting direct cast to int16 for VAD.")
                      audio_chunk_np = audio_chunk_np.astype(np.int16)

            audio_bytes = audio_chunk_np.tobytes()
            num_frames = len(audio_bytes) // self.vad_frame_bytes

            if num_frames == 0:
                 logger.debug("Audio chunk too short for a single VAD frame.")
                 return False # Cannot determine speech if shorter than a frame

            contains_speech = False
            for i in range(num_frames):
                frame_start = i * self.vad_frame_bytes
                frame_end = frame_start + self.vad_frame_bytes
                frame = audio_bytes[frame_start:frame_end]
                # Ensure frame has the correct length before passing to VAD
                if len(frame) == self.vad_frame_bytes:
                    if self.vad.is_speech(frame, self.vad_sample_rate):
                        contains_speech = True
                        break # Found speech, no need to check further
                else:
                     logger.warning(f"Skipping VAD frame {i+1}/{num_frames} due to incorrect length ({len(frame)} bytes, expected {self.vad_frame_bytes}).")


            logger.debug(f"VAD result for chunk: {'Speech detected' if contains_speech else 'No speech detected'}")
            return contains_speech

        except Exception as e:
            logger.error(f"Error during VAD processing: {e}", exc_info=True)
            return True # Assume speech on VAD error to avoid dropping actual speech

    def _audio_processing_thread(self):
        """Thread for continuous audio processing"""
        logger.info("Starting audio processing thread")

        # Start a conversation session (optional, depends on desired structure)
        # You might want to manage conversation sessions differently, e.g., based on silence duration
        current_conversation_id = self.db.start_conversation()
        if not current_conversation_id:
            logger.error("Failed to start initial conversation session. Audio processing cannot proceed.")
            return # Stop the thread if DB fails

        audio_config = self.config["audio"]
        chunk_duration = audio_config["chunk_duration"]
        # Remove device_id = audio_config["device_id"] - Handled by start_stream

        while self.running:
            try:
                # Record audio chunk from the stream queue
                # Remove device_id, channels, dtype arguments
                audio_chunk = self.stt.record_audio_chunk(
                    duration=chunk_duration
                    # dtype='int16' # dtype is set when starting the stream
                )

                if audio_chunk is None:
                    # Check if the stream is still supposed to be running
                    if not self.running or self.stt.stream_stop_event.is_set():
                         logger.info("Audio stream stopped or stop requested, exiting audio thread.")
                         break # Exit loop if stream stopped or stop requested
                    else:
                         logger.debug("No audio chunk received, but stream should be running. Waiting...")
                         time.sleep(0.1) # Wait briefly if queue was temporarily empty
                         continue

                # --- VAD Check ---
                if not self._is_speech(audio_chunk):
                    # Clear previous transcription if no speech detected now
                    with self.display_lock:
                        if self.latest_transcription:
                            self.latest_transcription = "" # Clear display
                    logger.debug("VAD detected no speech, skipping transcription.")
                    time.sleep(0.1) # Small sleep to prevent busy-looping on silence
                    continue # Skip processing if no speech detected
                # --- End VAD Check ---

                # Transcribe audio chunk
                # Ensure transcribe method can handle int16 or convert internally if needed
                transcription = self.stt.transcribe(audio_chunk)

                if not transcription or not transcription.get("text", "").strip():
                    logger.debug("Transcription result is empty.")
                    # Clear transcription display if needed
                    with self.display_lock:
                         if self.latest_transcription:
                              self.latest_transcription = "" # Clear display
                    continue

                # Get voice embedding if voice biometrics is enabled
                voice_embedding = None
                voice_identity = None
                if self.use_voice_biometrics and self.voice_biometrics:
                    # Ensure identify_speaker can handle int16 or convert internally
                    # --- Pass sample rate to identify_speaker --- 
                    voice_results = self.voice_biometrics.identify_speaker(
                        audio_chunk,
                        sample_rate=self.vad_sample_rate # Get sample rate from config/VAD settings
                    )
                    # --- End pass sample rate ---
                    if voice_results:
                        voice_embedding = voice_results.get("embedding") # Note: embedding might not be directly returned by new identify_speaker
                        voice_identity = voice_results # The result dict IS the identity

                # Get current video frame and detected faces for speaker matching
                face_identities = []
                lip_sync_scores = {} # Expecting a dict: {face_index_or_id: score}
                current_face_results = {} # Store results for display

                if self.use_video and self.video_capture and self.face_recognizer:
                    frame = self.video_capture.get_current_frame()

                    if frame is not None:
                        face_results = self.face_recognizer.identify_faces(frame)

                        if face_results:
                            current_face_results = face_results
                            face_identities = face_results.get("identities", [])
                            face_locations = face_results.get("face_locations", []) # Get locations

                            if self.use_lip_sync and self.lip_sync_detector and face_locations:
                                # --- Lip Sync Score Calculation ---
                                # Iterate through detected faces and calculate lip sync scores
                                for i, face_loc in enumerate(face_locations):
                                    # Ensure audio_chunk is in the correct format (e.g., float32) if needed by detector
                                    # You might need to convert audio_chunk (int16) here
                                    # Example conversion (adjust based on LipSyncDetector needs):
                                    # audio_chunk_float = audio_chunk.astype(np.float32) / 32768.0
                                    try:
                                        # Assuming detect_sync needs frame, face_location, and audio data
                                        # The key for lip_sync_scores should match how SpeakerMatcher expects it
                                        # Using index 'i' here as an example, adjust if needed (e.g., use face_id if available)
                                        score = self.lip_sync_detector.detect_sync(frame, face_loc, audio_chunk) # Pass the original int16 chunk or converted float
                                        if score is not None:
                                            lip_sync_scores[i] = score # Store score with face index as key
                                        else:
                                             lip_sync_scores[i] = 0.0 # Default score if detection fails
                                    except Exception as lip_e:
                                        logger.error(f"Error during lip sync detection for face {i}: {lip_e}", exc_info=False)
                                        lip_sync_scores[i] = 0.0 # Default score on error
                                # --- End Lip Sync Score Calculation ---

                # Determine who is speaking using speaker matching decision engine
                speaker_result = self.speaker_matcher.determine_speaker(
                    face_identities=face_identities,
                    lip_sync_scores=lip_sync_scores,
                    voice_identity=voice_identity
                )

                # --- Update shared state for display --- 
                with self.display_lock:
                    # Frame is updated by video thread, only update analysis results here
                    self.latest_face_results = current_face_results
                    # Store the dictionary of lip sync scores
                    self.latest_lip_sync_scores = lip_sync_scores # Store the dict
                    self.latest_speaker_result = speaker_result or {} # Ensure it's a dict
                    self.latest_transcription = transcription.get("text", "")
                # --- End update shared state ---

                # --- Store Utterance & Handle Voice Enroll/Enrich ---
                current_time = time.monotonic() # Get current time for cooldown checks
                speaker_external_id = "unknown"
                speaker_name = "Unknown"
                confidence = 0.0
                is_known_speaker = False

                if speaker_result:
                    speaker_external_id = speaker_result.get("speaker_id", "unknown")
                    confidence = speaker_result.get("confidence", 0.0)
                    speaker_name = speaker_result.get("name", speaker_external_id)
                    is_known_speaker = speaker_external_id != "unknown"
                    logger.info(f"Speaker: {speaker_name} ({speaker_external_id}), Confidence: {confidence:.2f}, Method: {speaker_result.get('method', 'N/A')}")
                else:
                    logger.info("Unknown speaker detected")

                logger.info(f"Said: {transcription.get('text', '')}")

                # Add utterance to DB
                utterance_id = self.db.add_utterance(
                    conversation_id=current_conversation_id,
                    text=transcription.get("text", ""),
                    speaker_external_id=speaker_external_id,
                    confidence=confidence,
                    # duration=... # Calculate duration if possible
                    # audio_path=... # Save audio chunk path if configured
                )
                if utterance_id is None:
                     logger.error(f"Failed to add utterance to conversation {current_conversation_id}")

                # --- Refined Voice Enrollment/Enrichment Logic ---
                if self.use_voice_biometrics and self.voice_biometrics and is_known_speaker:
                    # Scenario 1: Enroll - Speaker is known (e.g., by face), but voice wasn't matched.
                    should_enroll = self.voice_enroll_known_face_enabled and voice_identity is None
                    # Scenario 2: Enrich - Speaker is known AND voice was matched.
                    should_enrich = self.voice_enrichment_enabled and voice_identity is not None

                    if should_enroll or should_enrich:
                        last_attempt = self.last_voice_enroll_enrich_attempt_times.get(speaker_external_id, 0.0)
                        if current_time - last_attempt > self.voice_enroll_enrich_interval:
                            action = "Enrolling" if should_enroll else "Enriching"
                            logger.info(f"Attempting voice {action.lower()} for known speaker: {speaker_name} (ID: {speaker_external_id})")
                            # Ensure audio_chunk is passed correctly (numpy array)
                            update_success = self.voice_biometrics.add_voice_embedding(
                                person_id=speaker_external_id,
                                name=speaker_name,
                                audio_data_np=audio_chunk, # Pass the numpy array
                                sample_rate=self.vad_sample_rate # Pass the correct sample rate
                            )
                            if update_success:
                                logger.info(f"Successfully processed voice {action.lower()} data for {speaker_name}.")
                            else:
                                logger.warning(f"Failed to process voice {action.lower()} data for {speaker_name}.")
                            # Update timestamp regardless of success to prevent rapid retries on failure
                            self.last_voice_enroll_enrich_attempt_times[speaker_external_id] = current_time
                        # else: logger.debug(f"Skipping voice enroll/enrich for {speaker_external_id}, interval not met.")
                # --- End Refined Logic ---

            except Exception as e:
                logger.error(f"Error in audio processing thread: {e}", exc_info=True)
                # Clear results on error?
                with self.display_lock:
                    self.latest_speaker_result = {}
                    self.latest_transcription = ""
                time.sleep(1) # Avoid tight loop on error

        # End the conversation when the thread stops (optional)
        if current_conversation_id:
            self.db.end_conversation(current_conversation_id)
            logger.info(f"Ended conversation session {current_conversation_id}")

    def _video_processing_thread(self):
        """Thread for continuous video processing and automatic face enrollment/enrichment."""
        if not self.use_video or not self.video_capture or not self.face_recognizer:
            logger.warning("Video processing or face recognizer disabled. Cannot run video thread.")
            return

        logger.info("Starting video processing thread")
        fps = self.config["video"]["fps"]
        frame_time = 1.0 / fps if fps > 0 else 0.1 # Avoid division by zero

        while self.running:
            try:
                start_time = time.monotonic()
                # Capture frame
                success = self.video_capture.capture_frame()

                if not success:
                    logger.warning("Failed to capture video frame")
                    time.sleep(0.1)
                    continue

                current_frame = self.video_capture.get_current_frame()
                if current_frame is None:
                    continue

                # --- Update shared frame for display --- 
                with self.display_lock:
                    self.latest_frame = current_frame.copy()
                # --- End update shared frame ---

                # --- Automatic Enrollment/Enrichment ---
                current_time = time.monotonic()
                # Identify faces in the current frame
                # We need identities to know if they are known/unknown and their IDs
                face_results = self.face_recognizer.identify_faces(current_frame)

                if face_results:
                    face_locations = face_results.get("face_locations", [])
                    face_identities = face_results.get("identities", [])

                    for i, face_loc in enumerate(face_locations):
                        if i >= len(face_identities): continue # Should not happen, but safety check

                        identity = face_identities[i]
                        face_id = identity.get("id") if identity else None
                        face_name = identity.get("name") if identity else None
                        is_known = face_id is not None

                        # Use face_id if known, or a special key for unknown faces for cooldown purposes
                        # Use tuple (top, left) as a proxy key for unknown faces within a short interval
                        # This helps prevent multiple enroll attempts for the *same* unknown face in quick succession
                        unknown_face_key = None
                        if not is_known:
                            top, _, _, left = face_loc
                            unknown_face_key = f"unknown_{top}_{left}" # Simple key based on location
                        
                        lookup_key = face_id if is_known else unknown_face_key
                        if lookup_key is None: continue # Should not happen if face_loc exists

                        last_attempt_time = self.last_enroll_enrich_attempt_times.get(lookup_key, 0.0)

                        if current_time - last_attempt_time > self.enroll_enrich_interval:
                            if not is_known:
                                # --- Enroll Unknown Face ---
                                new_id = str(uuid.uuid4())
                                # Generate a temporary name, e.g., "Person_" + last 6 chars of ID
                                temp_name = f"Person_{new_id[-6:]}"
                                logger.info(f"Auto-enrolling new face detected. Assigning ID: {new_id}, Name: {temp_name}")

                                enroll_success = self.face_recognizer.add_face(
                                    person_id=new_id,
                                    name=temp_name,
                                    image_data=current_frame,
                                    face_location=face_loc,
                                    save_encoding=True # Save immediately
                                )

                                if enroll_success:
                                    logger.info(f"Successfully added face encoding for '{temp_name}' (ID: {new_id}).")
                                    # Add to database
                                    speaker_db_id = self.db.add_or_update_speaker(external_id=new_id, name=temp_name)
                                    if speaker_db_id:
                                         logger.info(f"Speaker '{temp_name}' (ID: {new_id}) added/updated in DB (DB ID: {speaker_db_id}).")
                                         # Update known face cooldown only on DB success
                                         self.last_enroll_enrich_attempt_times[new_id] = current_time
                                         # Remove the temporary unknown key cooldown
                                         self.last_enroll_enrich_attempt_times.pop(unknown_face_key, None)
                                    else:
                                         logger.error(f"Failed to add/update speaker '{temp_name}' (ID: {new_id}) in DB.")
                                         # Update unknown cooldown even on DB failure to prevent retrying same face loc
                                         self.last_enroll_enrich_attempt_times[unknown_face_key] = current_time
                                else:
                                    logger.error(f"Failed to add face encoding for '{temp_name}'.")
                                    # Update unknown cooldown to prevent rapid retries on encoding error
                                    self.last_enroll_enrich_attempt_times[unknown_face_key] = current_time

                            else:
                                # --- Enrich Known Face ---
                                logger.debug(f"Attempting to enrich data for known face: {face_name} (ID: {face_id})")
                                enrich_success = self.face_recognizer.add_face(
                                    person_id=face_id,
                                    name=face_name, # Pass existing name
                                    image_data=current_frame,
                                    face_location=face_loc,
                                    save_encoding=True # Save/update immediately
                                )
                                if enrich_success:
                                     logger.debug(f"Enrichment data processed for {face_name} (ID: {face_id}).")
                                     # Update timestamp on successful processing
                                     self.last_enroll_enrich_attempt_times[face_id] = current_time
                                else:
                                     logger.warning(f"Failed to process enrichment data for {face_name} (ID: {face_id}).")
                                     # Update timestamp even on failure to avoid rapid retries on processing error
                                     self.last_enroll_enrich_attempt_times[face_id] = current_time
                        # else: logger.debug(f"Skipping enroll/enrich for {lookup_key}, interval not met.") # Can be noisy

                # --- End Automatic Enrollment/Enrichment ---


                # Calculate time taken and sleep accordingly
                elapsed_time = time.monotonic() - start_time
                sleep_time = max(0, frame_time - elapsed_time)
                time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error in video processing thread: {e}", exc_info=True)
                with self.display_lock:
                    self.latest_frame = None # Clear frame on error
                time.sleep(1) # Avoid tight loop on error
    
    def start(self):
        """Start the background processing threads"""
        if self.running:
            logger.warning("System is already running")
            return False
        
        logger.info("Starting Who Dey Tallk background threads...")
        self.running = True
        self.threads = [] # Reset threads list

        # --- Start Audio Stream BEFORE starting threads that use it ---
        if not self.stt.start_stream(device_id=self.audio_device_id, channels=self.audio_channels, dtype=self.audio_dtype):
             logger.error("Failed to start audio stream. Aborting start.")
             self.running = False
             return False
        # --- End Start Audio Stream ---

        # Start video processing thread if video is enabled
        if self.use_video:
            video_thread = threading.Thread(
                target=self._video_processing_thread,
                daemon=True,
                name="VideoThread"
            )
            video_thread.start()
            self.threads.append(video_thread)
        
        # Start audio processing thread
        audio_thread = threading.Thread(
            target=self._audio_processing_thread,
            daemon=True,
            name="AudioThread"
        )
        audio_thread.start()
        self.threads.append(audio_thread)

        # DO NOT start monitoring thread here
        logger.info("Background threads started")
        return True
    
    def stop(self):
        """Stop the Who Dey Tallk system"""
        if not self.running:
            # logger.warning("System is not running") # Can be noisy if called after stop
            return
        
        logger.info("Stopping Who Dey Tallk system...")
        self.running = False # Signal threads to stop

        # --- Stop Audio Stream FIRST ---
        # This signals record_audio_chunk to stop waiting and allows the thread to exit
        if self.stt:
             logger.debug("Stopping audio stream...")
             self.stt.stop_stream()
        # --- End Stop Audio Stream ---

        # Stop monitoring UI if running (needs a way to break the mainloop)
        # Remove monitor stop if replacing
        # if self.monitor and hasattr(self.monitor, 'stop'):
        #      logger.debug("Stopping monitor UI...")
        #      self.monitor.stop() # Assuming monitor has a stop method that destroys the window

        # Close OpenCV window if it was opened
        if self.show_video_window:
            cv2.destroyAllWindows()

        # Wait for all background threads to finish
        logger.debug(f"Waiting for {len(self.threads)} background threads to join...")
        for thread in self.threads:
            if thread.is_alive():
                logger.debug(f"Joining thread: {thread.name}")
                thread.join(timeout=2.0)
                if thread.is_alive():
                     logger.warning(f"Thread {thread.name} did not exit cleanly.")

        # Clean up resources
        if self.video_capture:
            logger.debug("Releasing video capture...")
            self.video_capture.release()

        # --- Close STT resources (includes stream cleanup just in case) ---
        if self.stt:
             logger.debug("Closing SpeechToText resources...")
             self.stt.close()
        # --- End Close STT ---

        # Close database connection
        if self.db:
            logger.debug("Closing database connection...")
            self.db.close()

        self.threads = [] # Clear threads list
        logger.info("Who Dey Tallk system stopped")

    def _draw_overlays(self, frame, face_results, lip_sync_scores, speaker_result, transcription):
        """Draw analysis results onto the video frame."""
        if frame is None:
            return None

        # Draw face detection boxes and names/IDs
        face_locations = face_results.get("face_locations", [])
        face_identities = face_results.get("identities", [])
        for i, face_loc in enumerate(face_locations):
            # --- Corrected Unpacking and Point Calculation ---
            try:
                # Unpack assuming (x, y, w, h) format
                x_f, y_f, w_f, h_f = face_loc
                # Convert to integers for drawing
                x, y, w, h = int(x_f), int(y_f), int(w_f), int(h_f)
                # Calculate bottom-right corner
                x2 = x + w
                y2 = y + h
                # Basic validation after conversion
                if w <= 0 or h <= 0:
                    logger.warning(f"Skipping drawing invalid box dimensions: w={w}, h={h} from {face_loc}")
                    continue
            except (ValueError, TypeError, IndexError):
                logger.warning(f"Skipping drawing due to invalid face location format or values: {face_loc}")
                continue
            
            # Draw face box using correct points (top-left and bottom-right)
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
            # --- End Correction ---

            # Prepare text: ID/Name and Lip Sync
            name = "Unknown"
            face_id = None
            if i < len(face_identities):
                identity = face_identities[i]
                if identity:
                    name = identity.get("name", "Unknown")
                    face_id = identity.get("id")

            lip_sync_text = ""
            if self.use_lip_sync and lip_sync_scores:
                 # *** Use the associated score from the dictionary (assuming key is index 'i') ***
                 score = lip_sync_scores.get(i, None) # Get score by index
                 if score is not None:
                      # Use threshold from config
                      lip_sync_threshold = self.config.get("lip_sync", {}).get("threshold", 0.5)
                      lip_sync_status = "Sync" if score > lip_sync_threshold else "No Sync"
                      lip_sync_text = f" Lip: {lip_sync_status} ({score:.2f})"
                 else:
                      lip_sync_text = " Lip: N/A" # Score not available for this face index

            label = f"{name}{lip_sync_text}"
            # Adjust text position relative to the corrected box
            text_y = y - 10 if y - 10 > 10 else y + 15 # Position above box if possible
            cv2.putText(frame, label, (x + 6, text_y), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        # Draw overall speaker identification
        speaker_name = "Unknown Speaker"
        speaker_conf = 0.0
        if speaker_result:
            speaker_name = speaker_result.get("name", speaker_result.get("speaker_id", "Unknown"))
            speaker_conf = speaker_result.get("confidence", 0.0)
        speaker_text = f"Speaker: {speaker_name} ({speaker_conf:.2f})"
        cv2.putText(frame, speaker_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw transcription
        if transcription:
            # Wrap text if too long
            max_width = frame.shape[1] - 20
            font_scale = 0.6
            thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            y0 = frame.shape[0] - 30 # Start near bottom
            dy = 20 # Line height
            
            lines = []
            current_line = ""
            for word in transcription.split():
                test_line = f"{current_line} {word}".strip()
                (w, h), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
                if w > max_width:
                    lines.append(current_line)
                    current_line = word
                else:
                    current_line = test_line
            lines.append(current_line)
            
            # Draw lines from bottom up
            for i, line in enumerate(reversed(lines)):
                 y = y0 - i * dy
                 if y < 50: break # Avoid drawing over speaker text
                 cv2.putText(frame, line, (10, y), font, font_scale, (255, 255, 0), thickness)

        return frame

    def run(self):
        """Run the main application loop (OpenCV window or headless)."""
        # Decide whether to show OpenCV window or run headless/Tkinter
        # For now, prioritize OpenCV window if enabled
        use_opencv_window = self.show_video_window and self.use_video

        try:
            if self.start(): # Start background threads
                logger.info("System running. Press 'q' in video window or Ctrl+C in terminal to stop.")

                if use_opencv_window:
                    logger.info("Video window enabled. Displaying feed...")
                    window_name = "Who Dey Tallk - Live Feed"
                    while self.running:
                        # Get latest data thread-safely
                        with self.display_lock:
                            frame_copy = self.latest_frame.copy() if self.latest_frame is not None else None
                            face_results_copy = self.latest_face_results.copy()
                            lip_sync_scores_copy = self.latest_lip_sync_scores.copy()
                            speaker_result_copy = self.latest_speaker_result.copy()
                            transcription_copy = self.latest_transcription
                        
                        # Create display frame if possible
                        display_frame = None
                        if frame_copy is not None:
                            display_frame = self._draw_overlays(
                                frame_copy,
                                face_results_copy,
                                lip_sync_scores_copy,
                                speaker_result_copy,
                                transcription_copy
                            )
                        else:
                            # Create a blank frame if no video input
                            h = self.config["video"].get("height", 480)
                            w = self.config["video"].get("width", 640)
                            display_frame = np.zeros((h, w, 3), dtype=np.uint8)
                            cv2.putText(display_frame, "No video feed", (w//2 - 100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            # Still draw speaker/transcription if available
                            display_frame = self._draw_overlays(display_frame, {}, {}, speaker_result_copy, transcription_copy)

                        # Show the frame
                        cv2.imshow(window_name, display_frame)

                        # Handle window events and delay
                        key = cv2.waitKey(30) & 0xFF # ~33fps refresh, check for key press

                        # --- Check for 'q' to quit ---
                        if key == ord('q'):
                            logger.info("'q' key pressed. Stopping...")
                            self.running = False # Signal threads to stop
                            break

                        # --- Remove 'e' key check ---
                        # Remove: elif key == ord('e'): ... block ...
                        # --- End Remove ---

                        # Check if window was closed
                        try:
                             if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                                  logger.info("Video window closed by user. Stopping...")
                                  self.running = False
                                  break
                        except cv2.error:
                             # Window might already be destroyed during shutdown
                             logger.debug("Error checking window property, likely already closed.")
                             self.running = False
                             break

                else: # Run headless or with Tkinter monitor (original logic)
                    # Re-add Tkinter logic here if needed as an alternative
                    logger.info("Running in headless mode (no video window). Press Ctrl+C to stop.")
                    while self.running:
                        # Keep main thread alive, periodically check if still running
                        time.sleep(1.0)

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt. Stopping...")
        except Exception as e:
             logger.error(f"Unhandled exception in main run loop: {e}", exc_info=True)
        finally:
            self.stop()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Who Dey Tallk System - Conversation Detection and Speaker Identification")
    parser.add_argument("--config", type=str, default=None, help="Path to configuration file")
    parser.add_argument("--list-devices", action="store_true", help="List available video and audio devices")
    parser.add_argument("--add-face", type=str, help="Add a new face to the database (format: 'id,name,image_path')")
    parser.add_argument("--add-voice", type=str, help="Add a new voice to the database (format: 'id,name,audio_path')")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    return parser.parse_args()


def list_devices():
    """List available video and audio devices"""
    from logic.video_capture import VideoCapture
    from logic.speech_to_text import SpeechToText
    
    print("\n=== Available Devices ===")
    print("\nVideo Devices:")
    VideoCapture.list_devices()
    
    print("\nAudio Devices:")
    SpeechToText.list_audio_devices()
    

def main():
    """Main entry point"""
    args = parse_args()
    exit_code = 0
    system = None
    db_manager = None
    
    # Set log level to DEBUG if debug flag is set
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # List devices if requested
    if args.list_devices:
        list_devices()
        return 0
    
    try:
        # Load config needed for add-face/add-voice
        base_dir = Path(__file__).parent.resolve()
        config_path = args.config or base_dir / "config" / "config.yaml"
        if not Path(config_path).exists():
             logger.critical(f"Configuration file not found: {config_path}")
             sys.exit(1)
        config = ConfigLoader(config_path).load()

        # Initialize DB Manager early if needed for add-face/voice
        db_manager = DatabaseManager(config=config)
        
        # Add a new face if requested
        if args.add_face:
            from logic.face_recognition import FaceRecognizer
            try:
                face_id, face_name, image_path_str = args.add_face.split(',', 2)
                image_path = Path(image_path_str)
                if not image_path.is_absolute():
                     image_path = base_dir / image_path # Assume relative to project root if not absolute

                if not image_path.exists():
                     raise FileNotFoundError(f"Image file not found: {image_path}")

                # Use consistent model path logic from config
                face_rec_config = config.get("face_recognition", {})
                model_path_str = face_rec_config.get("model_path", "models/yolov8n.pt")
                model_full_path = base_dir / model_path_str
                known_faces_dir = base_dir / face_rec_config.get("known_faces_dir", "input/known_faces")

                recognizer = FaceRecognizer(
                     model_path=str(model_full_path), # Pass the specific model file path
                     known_faces_dir=str(known_faces_dir),
                     # Pass other necessary config if needed by add_face
                )
                # Add face to the recognizer system
                recognizer.add_face(face_id, face_name, str(image_path))
                logger.info(f"Successfully added face '{face_name}' (ID: {face_id}) to recognizer.")

                # Add/Update speaker in the database
                speaker_db_id = db_manager.add_or_update_speaker(external_id=face_id, name=face_name)
                if speaker_db_id:
                     logger.info(f"Successfully added/updated speaker '{face_name}' (ID: {face_id}) in the database (DB ID: {speaker_db_id}).")
                else:
                     logger.error(f"Failed to add/update speaker '{face_name}' (ID: {face_id}) in the database.")
                     exit_code = 1 # Indicate partial failure

            except ValueError:
                 logger.error("Invalid format for --add-face. Use 'id,name,image_path'.", exc_info=True)
                 exit_code = 1
            except FileNotFoundError as e:
                 logger.error(f"Error adding face: {e}", exc_info=True)
                 exit_code = 1
            except Exception as e:
                 logger.error(f"Unexpected error adding face: {e}", exc_info=True)
                 exit_code = 1
            finally:
                 if db_manager: db_manager.close() # Close DB if opened for this utility
                 sys.exit(exit_code)

        # Add a new voice if requested
        if args.add_voice:
            from logic.voice_biometrics import VoiceBiometricsEngine
            # --- Add librosa import --- 
            import librosa
            try:
                voice_id, voice_name, audio_path_str = args.add_voice.split(',', 2)
                audio_path = Path(audio_path_str)
                if not audio_path.is_absolute():
                     audio_path = base_dir / audio_path # Assume relative to project root

                if not audio_path.exists():
                     raise FileNotFoundError(f"Audio file not found: {audio_path}")

                # --- Load audio file using librosa --- 
                logger.info(f"Loading audio file: {audio_path}")
                # Use target sample rate from config, default to 16000
                target_sr = config.get("audio", {}).get("sample_rate", 16000)
                audio_data_np, loaded_sr = librosa.load(audio_path, sr=target_sr, mono=True)
                logger.info(f"Loaded audio with shape {audio_data_np.shape} and sample rate {loaded_sr} (target: {target_sr})")
                # --- End Load audio file ---

                # Use consistent config logic
                voice_bio_config = config.get("voice_biometrics", {})
                embeddings_path = base_dir / voice_bio_config.get("embeddings_path", "input/voice_embeddings")
                threshold = voice_bio_config.get("threshold", 0.85) # Use threshold from config

                engine = VoiceBiometricsEngine(
                    embeddings_path=str(embeddings_path),
                    threshold=threshold
                 )

                # --- Call add_voice_embedding with numpy array and sample rate ---
                add_success = engine.add_voice_embedding(
                    person_id=voice_id,
                    name=voice_name,
                    audio_data_np=audio_data_np, # Pass the loaded numpy array
                    sample_rate=loaded_sr # Pass the actual loaded sample rate
                )
                # --- End Call ---

                if add_success:
                    logger.info(f"Successfully processed and saved voice embedding for '{voice_name}' (ID: {voice_id}).")
                    # Add/Update speaker in the database
                    speaker_db_id = db_manager.add_or_update_speaker(external_id=voice_id, name=voice_name)
                    if speaker_db_id:
                        logger.info(f"Successfully added/updated speaker '{voice_name}' (ID: {voice_id}) in the database (DB ID: {speaker_db_id}).")
                    else:
                        logger.error(f"Failed to add/update speaker '{voice_name}' (ID: {voice_id}) in the database.")
                        exit_code = 1 # Indicate partial failure
                else:
                     logger.error(f"Failed to process/save voice embedding for '{voice_name}' (ID: {voice_id}).")
                     exit_code = 1 # Indicate failure

            except ValueError:
                 logger.error("Invalid format for --add-voice. Use 'id,name,audio_path'.", exc_info=True)
                 exit_code = 1
            except FileNotFoundError as e:
                 logger.error(f"Error adding voice: {e}", exc_info=True)
                 exit_code = 1
            except Exception as e:
                 logger.error(f"Unexpected error adding voice: {e}", exc_info=True)
                 exit_code = 1
            finally:
                 if db_manager: db_manager.close()
                 sys.exit(exit_code)

        # --- Run Main Application ---
        # If we haven't exited due to a utility argument, run the main system.
        logger.info("Initializing WhoDeyTallk system...")
        system = WhoDeyTallk(config_path=args.config)
        # Ensure the DB connection inside the system instance is used from now on
        db_manager = system.db # Use the instance managed by the system

        logger.info("Starting WhoDeyTallk system run loop...")
        system.run() # This blocks until stop is called or UI is closed
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Stopping...")
        exit_code = 0 # Clean exit on Ctrl+C
    except Exception as e:
        # Use logger if initialized, otherwise print
        log_func = logger.critical if 'logger' in globals() and isinstance(logger, logging.Logger) else print
        log_func(f"Critical error during Who Dey Tallk initialization or execution: {e}", exc_info=True)
        exit_code = 1
    finally:
        logger.info("Executing final cleanup...")
        if system and system.running:
             logger.info("Ensuring system shutdown in finally block...")
             system.stop() # This should handle DB closing internally now
        elif db_manager: # If only utility args were used, close the manually created DB manager
             logger.info("Closing standalone DB manager connection...")
             db_manager.close()

        if 'cv2' in sys.modules: # Check if cv2 was imported before trying to destroy windows
             cv2.destroyAllWindows()
        if 'logging' in sys.modules:
             logging.shutdown()

        logger.info(f"Who Dey Tallk finished with exit code {exit_code}.")
        return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)