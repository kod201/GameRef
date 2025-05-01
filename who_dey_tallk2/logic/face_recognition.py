#!/usr/bin/env python3
"""
Face Recognition module for Who Dey Tallk 2

Detects and recognizes faces in video frames
"""
import os
import cv2
import logging
import threading
import numpy as np
import pickle
from pathlib import Path
from ultralytics import YOLO  # Import YOLO
from deepface import DeepFace # Import DeepFace
from scipy.spatial import distance # For cosine distance calculation
import time # Import time for track cleanup

logger = logging.getLogger(__name__)

class FaceRecognizer:
    """
    Face detection and recognition class using YOLO for detection/tracking
    and DeepFace for recognition.
    """

    def __init__(self, model_path=None, known_faces_dir=None,
                 detection_threshold=0.5, recognition_threshold=0.5,
                 detect_every_n_frames=1, # Detection/Tracking frequency
                 deepface_model_name="VGG-Face",
                 deepface_detector_backend="skip",
                 tracker_config="bytetrack.yaml", # YOLO tracker config
                 track_memory_seconds=10): # How long to remember a track ID without seeing it
        """
        Initialize the face recognizer

        Args:
            model_path: Path to the YOLO face detection model (e.g., 'yolov8n-face.pt').
                        If None, defaults to 'models/yolov8n-face.pt' relative to project root.
            known_faces_dir: Directory containing known face images/encodings (for recognition).
            detection_threshold: Confidence threshold for YOLO face detection/tracking.
            recognition_threshold: Cosine distance threshold for face recognition (lower is better).
            detect_every_n_frames: Only detect/track faces every N frames (for performance).
            deepface_model_name: Name of the model DeepFace should use (e.g., "VGG-Face", "FaceNet", "ArcFace", "SFace").
            deepface_detector_backend: Backend for DeepFace's internal detection (use 'skip' if providing ROI).
            tracker_config (str): YOLO tracker configuration file (e.g., 'bytetrack.yaml', 'botsort.yaml').
            track_memory_seconds (int): Duration in seconds to keep track information for unseen IDs.
        """
        # If model_path is None, construct default path relative to this file's parent's parent
        if model_path is None:
             # Assumes models dir is at the same level as logic dir
             self.model_path = Path(__file__).parent.parent / "models" / "yolov8n-face.pt"
        else:
             self.model_path = Path(model_path)

        self.known_faces_dir = Path(known_faces_dir) if known_faces_dir else None
        self.detection_threshold = detection_threshold
        self.recognition_threshold = recognition_threshold # Now represents cosine distance threshold
        self.detect_every_n_frames = detect_every_n_frames # Now controls tracking frequency
        self.deepface_model_name = deepface_model_name
        self.deepface_detector_backend = deepface_detector_backend
        self.tracker_config = tracker_config
        self.track_memory_seconds = track_memory_seconds

        # Internal state
        self.face_detector = None  # This will hold the YOLO model
        self.known_face_encodings = {}  # {id: encoding} - For recognition part
        self.known_face_names = {}      # {id: name} - For recognition part
        self.lock = threading.RLock()   # Protect shared resources if used concurrently
        self.frame_count = 0
        # --- New state for tracking ---
        self.tracked_identities = {} # {track_id: {'id': ..., 'name': ..., 'confidence': ...}}
        self.last_seen_time = {}     # {track_id: timestamp}
        # --- End new state ---

        # Initialize models (loads YOLO detector)
        self._init_models()

        # Load known faces (recognition part - can be kept or removed if only detection is needed)
        if self.known_faces_dir:
            self.load_known_faces()

    def _init_models(self):
        """Initialize face detection model (YOLOv8)"""
        print("DEBUG: Attempting YOLOv8 initialization...") # Debug print
        try:
            if not self.model_path.exists():
                logger.error(f"YOLOv8 model file not found at {self.model_path}")
                # Attempt fallback or handle error appropriately
                # Trying relative path from project root as fallback
                alt_path = Path("models") / "yolov8n-face.pt"
                if alt_path.exists():
                    logger.warning(f"Model not found at {self.model_path}, trying {alt_path}")
                    self.model_path = alt_path
                else:
                    logger.error(f"YOLOv8 model file not found at {self.model_path} or {alt_path}")
                    self.face_detector = None
                    return

            # Load the YOLOv8 model
            self.face_detector = YOLO(str(self.model_path))
            logger.info(f"YOLOv8 Face detection model initialized from {self.model_path}")

        except ImportError:
             logger.error("ultralytics library not found. Please install it: pip install ultralytics")
             self.face_detector = None
        except Exception as e:
            logger.error(f"Error initializing YOLOv8 face model: {e}", exc_info=True)
            self.face_detector = None # Indicate failure

    def load_known_faces(self):
        """Load known faces from directory or pre-computed pickle file."""
        if not self.known_faces_dir or not self.known_faces_dir.exists():
            logger.warning(f"Known faces directory not found or not specified: {self.known_faces_dir}")
            return

        encodings_file = self.known_faces_dir / "encodings.pickle"
        regenerate_encodings = False

        try:
            if encodings_file.exists():
                logger.info(f"Loading pre-computed face encodings from {encodings_file}")
                with open(encodings_file, "rb") as f:
                    data = pickle.load(f)
                    # Check if the model used for saved encodings matches the current model
                    saved_model_name = data.get("model_name")
                    if saved_model_name == self.deepface_model_name:
                        self.known_face_encodings = data.get("encodings", {})
                        self.known_face_names = data.get("names", {})
                        logger.info(f"Loaded {len(self.known_face_encodings)} known faces from pickle (Model: {saved_model_name})")
                        # Check if any encoding is invalid (e.g., empty array)
                        invalid_encodings = {pid for pid, enc in self.known_face_encodings.items() if not isinstance(enc, np.ndarray) or enc.size == 0}
                        if invalid_encodings:
                            logger.warning(f"Found invalid encodings in pickle file for IDs: {invalid_encodings}. Recommend regenerating.")
                            # Optionally force regeneration: regenerate_encodings = True
                        return # Successfully loaded valid encodings
                    else:
                        logger.warning(f"Pickle file model ({saved_model_name}) does not match current model ({self.deepface_model_name}). Regenerating encodings.")
                        regenerate_encodings = True
            else:
                logger.info("No pre-computed encodings file found. Processing images.")
                regenerate_encodings = True

            if regenerate_encodings:
                self.known_face_encodings.clear()
                self.known_face_names.clear()
                logger.info(f"Processing known faces from {self.known_faces_dir} using model {self.deepface_model_name}")
                face_count = 0
                processed_ids = set()

                # Iterate through subdirectories (assuming one per person)
                for person_dir in self.known_faces_dir.iterdir():
                    if person_dir.is_dir():
                        person_id = person_dir.name
                        if person_id in processed_ids:
                            logger.warning(f"Duplicate person ID directory found: {person_id}. Skipping.")
                            continue

                        name_file = person_dir / "name.txt"
                        name = person_id # Default name to ID
                        if name_file.exists():
                            with open(name_file, "r") as f:
                                name = f.read().strip()

                        logger.debug(f"Processing images for {name} (ID: {person_id})")
                        person_encodings = []
                        image_files = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png")) + list(person_dir.glob("*.jpeg"))

                        if not image_files:
                            logger.warning(f"No image files found for person ID {person_id}")
                            continue

                        for img_path in image_files:
                            try:
                                # Use add_face logic to get encoding, but don't save individually
                                encoding = self._get_encoding_from_image_path(str(img_path))
                                if encoding is not None:
                                    person_encodings.append(encoding)
                            except Exception as e:
                                logger.error(f"Failed to process image {img_path} for {person_id}: {e}", exc_info=True)

                        if person_encodings:
                            # Average the encodings for this person (or use the first one if only one)
                            avg_encoding = np.mean(person_encodings, axis=0)
                            # Normalize the average encoding
                            norm = np.linalg.norm(avg_encoding)
                            if norm > 0:
                                avg_encoding = avg_encoding / norm
                                self.known_face_encodings[person_id] = avg_encoding.astype(np.float32) # Store as float32
                                self.known_face_names[person_id] = name
                                face_count += 1
                                processed_ids.add(person_id)
                                logger.debug(f"Generated encoding for {name} (ID: {person_id}) from {len(person_encodings)} images.")
                            else:
                                logger.warning(f"Could not normalize average encoding for {person_id}. Skipping.")
                        else:
                            logger.warning(f"Could not generate any valid encodings for {person_id} from images.")

                logger.info(f"Finished processing images. Generated encodings for {face_count} known faces.")
                if face_count > 0:
                    self._save_encodings() # Save the newly generated encodings

        except (pickle.UnpicklingError, EOFError) as pe:
             logger.error(f"Error loading pickle file {encodings_file}: {pe}. File might be corrupted. Regenerating.")
             # Force regeneration on next run or handle appropriately
             if encodings_file.exists():
                 try:
                     encodings_file.unlink() # Delete corrupted file
                     logger.info(f"Deleted corrupted pickle file: {encodings_file}")
                 except OSError as oe:
                     logger.error(f"Failed to delete corrupted pickle file {encodings_file}: {oe}")
             # Clear in-memory data
             self.known_face_encodings.clear()
             self.known_face_names.clear()
             # Optionally, trigger regeneration immediately
             # self.load_known_faces() # Be careful of recursion
        except Exception as e:
            logger.error(f"Error loading known faces: {e}", exc_info=True)
            self.known_face_encodings.clear() # Clear potentially partial data
            self.known_face_names.clear()

    def _save_encodings(self):
        """Save face encodings and current model name to disk"""
        if not self.known_faces_dir:
            logger.warning("Cannot save encodings: known_faces_dir not set.")
            return
        try:
            if not self.known_faces_dir.exists():
                self.known_faces_dir.mkdir(parents=True, exist_ok=True)

            encodings_file = self.known_faces_dir / "encodings.pickle"

            # Ensure all encodings are numpy arrays before saving
            valid_encodings = {pid: enc for pid, enc in self.known_face_encodings.items() if isinstance(enc, np.ndarray)}
            if len(valid_encodings) != len(self.known_face_encodings):
                logger.warning("Attempting to save non-numpy array encodings. Filtering invalid entries.")

            with open(encodings_file, "wb") as f:
                pickle.dump({
                    "encodings": valid_encodings,
                    "names": self.known_face_names,
                    "model_name": self.deepface_model_name # Store the model name
                }, f)

            logger.info(f"Saved {len(valid_encodings)} face encodings to {encodings_file} (Model: {self.deepface_model_name})")

        except Exception as e:
            logger.error(f"Error saving face encodings: {e}", exc_info=True)

    def compute_face_encoding(self, frame, face_location):
        """
        Compute face encoding for recognition using DeepFace.

        Args:
            frame: Input image frame (BGR format).
            face_location: Face location as (x, y, w, h) tuple.

        Returns:
            numpy.ndarray: Normalized face embedding vector (float32) or None if failed.
        """
        try:
            # Extract the face ROI
            x, y, w, h = map(int, face_location) # Ensure integer coordinates

            # Add basic validation for ROI coordinates
            if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                 logger.warning(f"Invalid face location provided for encoding: {face_location}, frame shape: {frame.shape}")
                 return None

            # Use cv2.getRectSubPix for potentially safer ROI extraction near boundaries
            center = (float(x + w / 2), float(y + h / 2))
            size = (int(w), int(h))
            face_img = cv2.getRectSubPix(frame, size, center)

            # Check if ROI extraction was successful
            if face_img is None or face_img.size == 0:
                logger.warning(f"Empty face ROI extracted for location {face_location} using getRectSubPix.")
                # Fallback to direct slicing (might be less safe but worth trying)
                y1, y2 = max(0, y), min(frame.shape[0], y + h)
                x1, x2 = max(0, x), min(frame.shape[1], x + w)
                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    logger.error(f"Empty face ROI even after fallback slicing for location {face_location}.")
                    return None

            # DeepFace expects BGR numpy array.
            # Use detector_backend='skip' as we provide the ROI directly.
            # enforce_detection=False prevents DeepFace from trying to detect again.
            embedding_objs = DeepFace.represent(
                img_path=face_img,
                model_name=self.deepface_model_name,
                detector_backend=self.deepface_detector_backend,
                enforce_detection=False,
                align=True # Perform facial alignment
            )

            # DeepFace.represent returns a list of dictionaries
            if embedding_objs and isinstance(embedding_objs, list) and 'embedding' in embedding_objs[0]:
                embedding = embedding_objs[0]['embedding']
                # Ensure it's a numpy array
                if isinstance(embedding, list):
                    embedding = np.array(embedding, dtype=np.float32)
                elif isinstance(embedding, np.ndarray):
                     embedding = embedding.astype(np.float32) # Ensure correct dtype
                else:
                    logger.warning(f"DeepFace returned unexpected embedding type: {type(embedding)}")
                    return None

                # Normalize the embedding (important for cosine distance)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                    return embedding
                else:
                    logger.warning(f"DeepFace returned zero vector embedding for face at {face_location}")
                    return None # Or return the zero vector if that's meaningful
            else:
                # This often happens if align=True and alignment fails
                logger.warning(f"DeepFace could not generate embedding for face at {face_location}. Alignment might have failed.")
                return None

        except ValueError as ve:
             # Catch specific DeepFace errors like "Face could not be detected" if enforce_detection was True
             # Or shape mismatches if alignment fails badly
             logger.warning(f"DeepFace ValueError during encoding: {ve} for location {face_location}")
             return None
        except Exception as e:
            # Catch potential errors from cv2.getRectSubPix or other issues
            logger.error(f"Error computing DeepFace encoding: {e}", exc_info=True)
            return None

    def identify_faces(self, frame):
        """
        Detect, track, and identify faces in a frame. Uses YOLO tracking and
        DeepFace embeddings. If a new track doesn't match known faces,
        it's automatically enrolled.

        Args:
            frame: Input image frame.

        Returns:
            dict: Dictionary with face locations and identities based on tracking.
                {
                    'face_locations': list of (x, y, w, h) tuples,
                    'identities': list of {'id', 'name', 'confidence', 'track_id'} dicts
                }
        """
        current_time = time.time()
        final_results = {'face_locations': [], 'identities': []}
        active_track_ids = set()

        if self.face_detector is None:
            logger.warning("YOLO detector not initialized. Cannot identify faces.")
            return final_results

        try:
            # Use a copy of the frame for safety if modifications happen elsewhere
            frame_copy = frame.copy()
            with self.lock: # Acquire lock for thread safety
                self.frame_count += 1
                # Skip tracking on some frames for performance
                if self.detect_every_n_frames > 1 and self.frame_count % self.detect_every_n_frames != 0:
                    self._cleanup_old_tracks(current_time)
                    # On skipped frames, return empty for now. Could be enhanced.
                    return final_results

                # Perform tracking with YOLOv8
                # Ensure frame is contiguous C-order array, which YOLO might prefer
                if not frame_copy.flags['C_CONTIGUOUS']:
                    frame_copy = np.ascontiguousarray(frame_copy)

                results = self.face_detector.track(frame_copy, conf=self.detection_threshold, persist=True, tracker=self.tracker_config, verbose=False)

                processed_locations = {} # track_id -> location
                processed_identities = {} # track_id -> identity dict

                # Process tracked faces
                if results and results[0].boxes and results[0].boxes.id is not None:
                    # Get boxes in xywh format (center_x, center_y, width, height) - KEEP AS FLOATS
                    tracked_boxes_xywh_center = results[0].boxes.xywh.cpu().numpy() # Removed .astype(int)
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)

                    for box_xywh_center, track_id in zip(tracked_boxes_xywh_center, track_ids):
                        active_track_ids.add(track_id)
                        self.last_seen_time[track_id] = current_time # Update last seen time

                        # Convert xywh (center) to xywh (top-left) using float arithmetic
                        cx, cy, w, h = box_xywh_center
                        x = cx - w / 2.0 # Use float division
                        y = cy - h / 2.0 # Use float division
                        face_location = (x, y, w, h) # Store as floats

                        # Ensure location is valid before processing (including frame boundaries)
                        # Check against frame shape using potentially float coords
                        frame_h, frame_w = frame_copy.shape[:2]
                        if not (w > 0 and h > 0 and x >= 0 and y >= 0 and (x + w) <= frame_w and (y + h) <= frame_h):
                            logger.warning(f"Skipping invalid face location derived from tracking: {face_location} for track ID {track_id}")
                            continue # Skip this box, don't try to identify

                        processed_locations[track_id] = face_location

                        # --- Check if track ID is already known ---
                        if track_id in self.tracked_identities:
                            identity = self.tracked_identities[track_id].copy() # Use copy
                            identity['track_id'] = track_id
                            processed_identities[track_id] = identity
                            # logger.debug(f"Reusing identity {identity['name']} for track ID {track_id}")
                            continue # Move to next tracked face

                        # --- New track ID: Perform recognition or enrollment ---
                        logger.info(f"New track ID {track_id} detected. Performing recognition/enrollment...")
                        # Pass the potentially modified frame_copy
                        encoding = self.compute_face_encoding(frame_copy, face_location)

                        current_identity = {
                            'id': None,
                            'name': 'Unknown',
                            'confidence': 0.0,
                            'track_id': track_id
                        }

                        if encoding is not None and self.known_face_encodings:
                            min_cosine_distance = float('inf')
                            best_match_id = None
                            best_match_name = 'Unknown'

                            for person_id, known_encoding in self.known_face_encodings.items():
                                # Ensure known_encoding is valid before comparison
                                if isinstance(known_encoding, np.ndarray) and known_encoding.size > 0:
                                    # Ensure encoding dimensions match before calculating distance
                                    if encoding.shape == known_encoding.shape:
                                        try:
                                            dist = distance.cosine(encoding, known_encoding) # Already float32
                                            if dist < min_cosine_distance:
                                                min_cosine_distance = dist
                                                best_match_id = person_id
                                                best_match_name = self.known_face_names.get(person_id, person_id)
                                        except Exception as dist_e: # Catch any error during distance calc
                                            logger.error(f"Error calculating distance for track {track_id} and known ID {person_id}: {dist_e}")
                                            min_cosine_distance = float('inf') # Ensure no match if error
                                            break # Stop comparing this face
                                    else:
                                        # This should ideally not happen if load_known_faces ensures consistency
                                        logger.error(f"Shape mismatch! New encoding shape {encoding.shape} vs known encoding shape {known_encoding.shape} for ID {person_id}. Skipping comparison.")
                                        # Don't break, maybe other known faces have correct shape
                                        continue
                                else:
                                    logger.warning(f"Skipping invalid known encoding for ID {person_id} during comparison for track {track_id}")


                            # --- Decision: Recognize Known or Enroll New ---
                            if min_cosine_distance != float('inf') and min_cosine_distance < self.recognition_threshold:
                                # **Recognized as Known Face**
                                confidence = max(0.0, 1.0 - (min_cosine_distance / self.recognition_threshold))
                                current_identity = {
                                    'id': best_match_id,
                                    'name': best_match_name,
                                    'confidence': confidence,
                                    'track_id': track_id
                                }
                                logger.info(f"Recognized track ID {track_id} as {best_match_name} (Conf: {confidence:.2f}, Dist: {min_cosine_distance:.3f})")
                            else:
                                # **Not Recognized as Known Face - Enroll Automatically**
                                if min_cosine_distance != float('inf'):
                                     logger.info(f"Track ID {track_id} determined as Unknown (Min dist: {min_cosine_distance:.3f} >= Threshold: {self.recognition_threshold:.3f}). Attempting enrollment.")
                                else: # Comparison failed or encoding was None initially
                                     logger.info(f"Track ID {track_id} could not be compared to known faces. Attempting enrollment.")

                                # Generate new ID and Name (simple example)
                                new_person_id = f"auto_{track_id}"
                                new_name = f"Person {track_id}"

                                logger.info(f"Enrolling new face for track ID {track_id} as ID: {new_person_id}, Name: {new_name}")

                                # Call add_face to enroll (uses the same lock implicitly)
                                # Pass frame_copy and face_location
                                success = self.add_face(
                                    person_id=new_person_id,
                                    name=new_name,
                                    image_data=frame_copy, # Use the current frame
                                    face_location=face_location, # Use the detected location
                                    save_encoding=True # Save immediately
                                )

                                if success:
                                    logger.info(f"Successfully enrolled new face for track ID {track_id} as {new_name}")
                                    # Update current identity to reflect the newly enrolled person
                                    current_identity = {
                                        'id': new_person_id,
                                        'name': new_name,
                                        'confidence': 1.0, # Confidence is high as it's just enrolled
                                        'track_id': track_id
                                    }
                                else:
                                    logger.error(f"Failed to enroll new face for track ID {track_id}. Marking as Unknown.")
                                    # Keep current_identity as Unknown

                        elif encoding is None:
                             logger.warning(f"Could not compute encoding for new track ID {track_id}. Cannot enroll. Marking as Unknown.")
                             # Keep current_identity as Unknown
                        else: # No known faces loaded initially
                            logger.warning(f"No known faces loaded. Attempting enrollment for new track ID {track_id}.")
                            # Enroll automatically (similar logic as above)
                            new_person_id = f"auto_{track_id}"
                            new_name = f"Person {track_id}"
                            logger.info(f"Enrolling first face for track ID {track_id} as ID: {new_person_id}, Name: {new_name}")
                            success = self.add_face(
                                person_id=new_person_id,
                                name=new_name,
                                image_data=frame_copy,
                                face_location=face_location,
                                save_encoding=True
                            )
                            if success:
                                logger.info(f"Successfully enrolled first face for track ID {track_id} as {new_name}")
                                current_identity = {
                                    'id': new_person_id,
                                    'name': new_name,
                                    'confidence': 1.0,
                                    'track_id': track_id
                                }
                            else:
                                logger.error(f"Failed to enroll first face for track ID {track_id}. Marking as Unknown.")
                                # Keep current_identity as Unknown


                        # Store the determined identity for this track ID
                        self.tracked_identities[track_id] = current_identity
                        processed_identities[track_id] = current_identity

                # --- Build final lists in correct order based on processed_locations ---
                for track_id, location in processed_locations.items():
                    final_results['face_locations'].append(location)
                    # Identity should always be present in processed_identities if location is valid
                    if track_id in processed_identities:
                        final_results['identities'].append(processed_identities[track_id])
                    else:
                         logger.error(f"Identity missing for processed track ID {track_id}. Appending Error identity.")
                         final_results['identities'].append({'id': None, 'name': 'Error', 'confidence': 0.0, 'track_id': track_id})


                # --- Cleanup old tracks ---
                self._cleanup_old_tracks(current_time, active_track_ids)

                return final_results

        except Exception as e:
            logger.error(f"Error identifying/tracking faces: {e}", exc_info=True)
            # Attempt to cleanup state on error? Maybe not necessary if transient.
            # self.tracked_identities.clear()
            # self.last_seen_time.clear()
            return {'face_locations': [], 'identities': []} # Return empty on error

    def _cleanup_old_tracks(self, current_time, active_ids=None):
        """Remove track IDs that haven't been seen for a while."""
        ids_to_remove = []
        # Use list() to avoid RuntimeError: dictionary changed size during iteration
        for track_id, last_seen in list(self.last_seen_time.items()):
            is_stale = (current_time - last_seen) > self.track_memory_seconds
            if is_stale:
                ids_to_remove.append(track_id)

        if ids_to_remove:
            removed_count = 0
            for track_id in ids_to_remove:
                # Use pop with default to avoid KeyError if already removed
                if self.tracked_identities.pop(track_id, None) is not None:
                    removed_count += 1
                self.last_seen_time.pop(track_id, None)

            if removed_count > 0:
                logger.debug(f"Cleaned up {removed_count} stale track IDs: {ids_to_remove}")

    def _get_encoding_from_image_path(self, image_path_str):
        """Helper to load image, detect largest face, and compute encoding."""
        image = cv2.imread(image_path_str)
        if image is None:
            logger.error(f"Failed to load image: {image_path_str}")
            return None

        # Temporarily disable frame skipping for single image detection
        original_skip = self.detect_every_n_frames
        self.detect_every_n_frames = 1
        # Use predict for single image detection (not tracking)
        results = self.face_detector.predict(image, conf=self.detection_threshold, verbose=False)
        self.detect_every_n_frames = original_skip # Restore setting

        face_locations_xyxy = []
        if results and results[0].boxes:
             for box in results[0].boxes:
                 x1, y1, x2, y2 = map(int, box.xyxy[0])
                 w = x2 - x1
                 h = y2 - y1
                 if w > 0 and h > 0:
                     face_locations_xyxy.append((x1, y1, x2, y2)) # Store xyxy

        if not face_locations_xyxy:
            logger.warning(f"No face detected in image: {image_path_str}")
            return None

        # Use the largest detected face (heuristic based on area)
        def area(box):
            x1, y1, x2, y2 = box
            return (x2 - x1) * (y2 - y1)

        best_box_xyxy = max(face_locations_xyxy, key=area)
        x1, y1, x2, y2 = best_box_xyxy
        face_location_xywh = (x1, y1, x2 - x1, y2 - y1) # Convert to xywh

        # Compute face encoding using the main method
        encoding = self.compute_face_encoding(image, face_location_xywh)
        if encoding is None:
            logger.warning(f"Failed to compute face encoding for image: {image_path_str}")
            return None

        return encoding

    def add_face(self, person_id, name, image_data=None, face_location=None, image_path=None, save_encoding=True):
        """
        Add a new face to the known faces. Can accept image data + location,
        or an image path. Computes encoding using DeepFace and saves.

        Args:
            person_id: Unique identifier for the person.
            name: Name of the person.
            image_data: Image frame (numpy array) containing the face. Required if image_path is None.
            face_location: (x, y, w, h) tuple for the face in image_data. Required if image_path is None.
            image_path: Path to the person's face image. Used if image_data/face_location are None.
            save_encoding (bool): If True, save the updated encodings to the pickle file immediately.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            encoding = None
            image_to_save = None # Store the image data used for saving ROI
            face_location_to_save = None # Store the location used

            if image_data is not None and face_location is not None:
                # --- Process provided image data and location ---
                image_to_save = image_data # Use provided data for saving ROI later
                face_location_to_save = face_location
                # Compute face encoding using DeepFace
                encoding = self.compute_face_encoding(image_data, face_location)
                if encoding is None:
                    logger.error(f"Failed to compute face encoding from provided image data for ID {person_id}")
                    return False

            elif image_path is not None:
                # --- Process image from path ---
                encoding = self._get_encoding_from_image_path(str(image_path))
                if encoding is None:
                    # Error already logged in helper function
                    return False
                # For saving ROI later, need to reload image and find location again (or modify helper)
                # For simplicity, we might skip saving ROI if only path is given, or accept the overhead.
                # Let's accept the overhead for now:
                image_to_save = cv2.imread(str(image_path))
                # Re-detect to get location (could optimize this)
                results = self.face_detector.predict(image_to_save, conf=self.detection_threshold, verbose=False)
                face_locations_xyxy = []
                if results and results[0].boxes:
                     for box in results[0].boxes:
                         x1, y1, x2, y2 = map(int, box.xyxy[0])
                         w = x2 - x1
                         h = y2 - y1
                         if w > 0 and h > 0:
                             face_locations_xyxy.append((x1, y1, x2, y2))
                if face_locations_xyxy:
                     def area(box): return (box[2] - box[0]) * (box[3] - box[1])
                     best_box_xyxy = max(face_locations_xyxy, key=area)
                     x1, y1, x2, y2 = best_box_xyxy
                     face_location_to_save = (x1, y1, x2 - x1, y2 - y1) # xywh
                else:
                    logger.warning(f"Could not re-detect face in {image_path} to save ROI.")
                    face_location_to_save = None # Cannot save ROI

            else:
                logger.error("add_face requires either (image_data and face_location) or image_path.")
                return False

            # --- Add to known faces in memory ---
            with self.lock:
                # Store encoding as float32
                self.known_face_encodings[person_id] = encoding.astype(np.float32)
                self.known_face_names[person_id] = name

                # --- Save image and update encodings file if known_faces_dir is set ---
                if self.known_faces_dir:
                    person_dir = self.known_faces_dir / person_id
                    person_dir.mkdir(parents=True, exist_ok=True)

                    # Save name
                    name_file = person_dir / "name.txt"
                    with open(name_file, "w") as f:
                        f.write(name)

                    # Save a copy of the face ROI (optional, good for reference)
                    if image_to_save is not None and face_location_to_save is not None:
                        x, y, w, h = map(int, face_location_to_save)
                        # Ensure ROI coordinates are valid within the image_to_save dimensions
                        y1, y2 = max(0, y), min(image_to_save.shape[0], y + h)
                        x1, x2 = max(0, x), min(image_to_save.shape[1], x + w)
                        face_roi = image_to_save[y1:y2, x1:x2]

                        if face_roi.size > 0:
                             # Use a consistent naming convention, maybe add timestamp or counter?
                             img_count = len(list(person_dir.glob(f"{person_id}_face_*.jpg")))
                             img_save_path = person_dir / f"{person_id}_face_{img_count + 1}.jpg"
                             cv2.imwrite(str(img_save_path), face_roi)
                             logger.debug(f"Saved face ROI to {img_save_path}")
                        else:
                             logger.warning(f"Extracted face ROI was empty for {person_id}. Cannot save ROI image.")
                    else:
                        logger.warning(f"Cannot save face ROI for {person_id}: image_to_save or face_location_to_save missing/invalid.")


                    # Update the persistent encodings file if requested
                    if save_encoding:
                        self._save_encodings()
                    else:
                        logger.debug(f"Encoding for {name} (ID: {person_id}) added to memory, but not saved to file yet (save_encoding=False).")


                logger.info(f"Processed and added face data for {name} (ID: {person_id})")
                return True

        except Exception as e:
            logger.error(f"Error adding/processing face: {e}", exc_info=True)
            return False


    def annotate_frame(self, frame, face_results):
        """
        Draw face boxes and labels on a frame based on detection/identification results.

        Args:
            frame: Input image frame.
            face_results: Results dictionary from identify_faces().

        Returns:
            numpy.ndarray: Annotated frame.
        """
        try:
            result = frame.copy() # Work on a copy

            face_locations = face_results.get('face_locations', [])
            identities = face_results.get('identities', [])

            # Ensure we have the same number of locations and identities
            num_faces = min(len(face_locations), len(identities))

            for i in range(num_faces):
                face_location = face_locations[i]
                identity = identities[i]

                # Ensure face_location is valid tuple/list of 4 ints/floats
                if not (isinstance(face_location, (tuple, list)) and len(face_location) == 4):
                    logger.warning(f"Skipping invalid face location format: {face_location}")
                    continue
                try:
                    # Convert to int for drawing, handle potential float values from tracker
                    x, y, w, h = map(int, face_location)
                except (ValueError, TypeError):
                    logger.warning(f"Skipping invalid coordinate types in face location: {face_location}")
                    continue

                # Basic check for valid box dimensions after conversion
                if w <= 0 or h <= 0:
                    logger.warning(f"Skipping invalid box dimensions after int conversion: w={w}, h={h}")
                    continue

                # Determine box color and label based on identification
                is_known = identity.get('id') is not None and identity.get('name') != 'Unknown'
                color = (0, 255, 0) if is_known else (0, 0, 255) # Green for known, Red for unknown

                # Draw face bounding box
                cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

                # Prepare label text
                name = identity.get('name', 'Error')
                confidence = identity.get('confidence', 0.0)
                track_id = identity.get('track_id', -1) # Get track ID
                label = f"ID:{track_id} {name}" # Include track ID in label
                if is_known:
                    label += f" ({confidence:.2f})"

                # Calculate text size for background rectangle
                (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                # Draw background rectangle for label
                # Position it slightly above the top-left corner of the face box
                label_y = max(y - 10, label_h + 10) # Ensure label doesn't go off-screen top
                # Ensure background doesn't go off-screen left/right
                label_x1 = max(0, x)
                label_x2 = min(result.shape[1], x + label_w) # Clamp width to frame boundary
                bg_y1 = max(0, label_y - label_h - baseline)
                bg_y2 = max(0, label_y)

                # Adjust label_w if clamped
                actual_label_w = label_x2 - label_x1

                cv2.rectangle(result, (label_x1, bg_y1), (label_x1 + actual_label_w, bg_y2), color, cv2.FILLED)

                # Draw label text (black text on colored background)
                # Position text relative to background box
                text_y = bg_y2 - 5 # Position near bottom of background
                cv2.putText(result, label, (label_x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            return result

        except Exception as e:
            logger.error(f"Error annotating frame: {e}", exc_info=True)
            return frame # Return original frame on error
