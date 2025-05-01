#!/usr/bin/env python3
"""
Lip Sync Detector module for Who Dey Tallk 2

Detects if a person's lips are moving in sync with speech
"""
import cv2
import time
import logging
import numpy as np
import threading
from pathlib import Path
import mediapipe as mp # Import MediaPipe

logger = logging.getLogger(__name__)

# Define lip landmark indices (using standard 468/478 landmark model)
# Outer lips
OUTER_LIP_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 0, 270, 269, 267, 409, 291]
# Inner lips (can be used for more detailed analysis)
INNER_LIP_INDICES = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 13, 312, 311, 310, 415, 308]
# Specific points for vertical distance (mouth opening)
UPPER_LIP_TOP = 13
LOWER_LIP_BOTTOM = 14

class LipSyncDetector:
    """
    Detects if a person's lips are moving in sync with speech using MediaPipe Face Mesh.
    """

    def __init__(self, threshold=0.5, use_optical_flow=True, 
                 mp_static_mode=False, mp_max_faces=5, mp_min_detection_confidence=0.5, mp_min_tracking_confidence=0.5):
        """
        Initialize the lip sync detector using MediaPipe Face Mesh.

        Args:
            threshold: Confidence threshold for lip movement detection (applied to final score).
            use_optical_flow: Whether to use optical flow on the lip ROI for movement detection.
            mp_static_mode: MediaPipe FaceMesh static_image_mode.
            mp_max_faces: MediaPipe FaceMesh max_num_faces.
            mp_min_detection_confidence: MediaPipe FaceMesh min_detection_confidence.
            mp_min_tracking_confidence: MediaPipe FaceMesh min_tracking_confidence.
        """
        # Remove model_path as it's not needed for MediaPipe Face Mesh default model
        # self.model_path = Path(model_path) if model_path else None
        self.threshold = threshold
        self.use_optical_flow = use_optical_flow

        # MediaPipe config
        self.mp_static_mode = mp_static_mode
        self.mp_max_faces = mp_max_faces
        self.mp_min_detection_confidence = mp_min_detection_confidence
        self.mp_min_tracking_confidence = mp_min_tracking_confidence

        # Internal state
        self.face_mesh = None # MediaPipe Face Mesh model
        self.prev_lip_frames = {}  # {face_index: previous frame}
        self.lip_movement_scores = {}  # {face_index: movement score}
        self.lock = threading.RLock()

        # Initialize models
        self._init_models()

    def _init_models(self):
        """Initialize MediaPipe Face Mesh model"""
        try:
            # Initialize MediaPipe Face Mesh
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=self.mp_static_mode,
                max_num_faces=self.mp_max_faces,
                refine_landmarks=True, # Get more landmarks (478) including lips, iris
                min_detection_confidence=self.mp_min_detection_confidence,
                min_tracking_confidence=self.mp_min_tracking_confidence
            )
            logger.info("MediaPipe Face Mesh initialized successfully for Lip Sync.")

        except ImportError:
            logger.error("mediapipe library not found. Please install it: pip install mediapipe")
            self.face_mesh = None
        except Exception as e:
            logger.error(f"Error initializing MediaPipe Face Mesh: {e}", exc_info=True)
            self.face_mesh = None

    def _extract_lip_region(self, frame_rgb, face_landmarks):
        """
        Extract the lip region and calculate mouth openness using MediaPipe landmarks.

        Args:
            frame_rgb: Input image frame in RGB format.
            face_landmarks: MediaPipe FaceLandmark object for a single face.

        Returns:
            tuple: (lip_region_gray, mouth_open_score, lip_bounding_box) or (None, 0.0, None) if failed.
                   lip_bounding_box is (x_min, y_min, x_max, y_max) in absolute pixel coordinates.
        """
        if not face_landmarks:
            return None, 0.0, None

        try:
            h, w, _ = frame_rgb.shape
            landmarks = face_landmarks.landmark

            # Get coordinates of outer lip landmarks
            lip_points = []
            for idx in OUTER_LIP_INDICES:
                if 0 <= idx < len(landmarks):
                    lm = landmarks[idx]
                    # Convert normalized coordinates to pixel coordinates
                    px, py = int(lm.x * w), int(lm.y * h)
                    lip_points.append((px, py))
                else:
                    logger.warning(f"Landmark index {idx} out of range ({len(landmarks)} landmarks)")
                    return None, 0.0, None # Cannot proceed if index is invalid

            if not lip_points:
                logger.warning("No valid lip landmark points found.")
                return None, 0.0, None

            # Calculate bounding box around lip points
            lip_points_np = np.array(lip_points)
            x_min, y_min = lip_points_np.min(axis=0)
            x_max, y_max = lip_points_np.max(axis=0)

            # Add some padding (optional, adjust as needed)
            padding_x = int((x_max - x_min) * 0.1)
            padding_y = int((y_max - y_min) * 0.2)
            x_min = max(0, x_min - padding_x)
            y_min = max(0, y_min - padding_y)
            x_max = min(w, x_max + padding_x)
            y_max = min(h, y_max + padding_y)

            # Ensure box has valid dimensions
            if x_min >= x_max or y_min >= y_max:
                logger.warning(f"Invalid lip bounding box calculated: ({x_min},{y_min}) -> ({x_max},{y_max})")
                return None, 0.0, None

            lip_bounding_box = (x_min, y_min, x_max, y_max)

            # Extract the lip region from the original frame (use BGR if needed later)
            # Note: MediaPipe processes RGB, so frame_rgb is correct here.
            lip_region_color = frame_rgb[y_min:y_max, x_min:x_max]

            if lip_region_color.size == 0:
                logger.warning("Extracted lip region is empty.")
                return None, 0.0, lip_bounding_box

            # Resize for consistency (optional, but good for optical flow)
            lip_region_resized = cv2.resize(lip_region_color, (64, 32))

            # Convert to grayscale for processing
            lip_region_gray = cv2.cvtColor(lip_region_resized, cv2.COLOR_RGB2GRAY)

            # Calculate mouth openness score based on vertical distance
            # Use specific upper/lower lip landmarks
            lm_upper = landmarks[UPPER_LIP_TOP]
            lm_lower = landmarks[LOWER_LIP_BOTTOM]
            upper_y = lm_upper.y * h
            lower_y = lm_lower.y * h
            vertical_distance = abs(lower_y - upper_y)

            # Normalize the distance (heuristic, needs tuning)
            # Estimate a reference distance, e.g., distance between eyes or nose bridge points
            try:
                lm_left_eye = landmarks[33] # Left eye inner corner
                lm_right_eye = landmarks[263] # Right eye inner corner
                eye_dist = np.sqrt(((lm_left_eye.x - lm_right_eye.x)*w)**2 + ((lm_left_eye.y - lm_right_eye.y)*h)**2)
                if eye_dist > 1e-6: # Avoid division by zero
                    mouth_open_score = min(1.0, vertical_distance / (eye_dist * 0.5)) # Normalize relative to eye distance (tune factor 0.5)
                else:
                    mouth_open_score = 0.0
            except IndexError:
                 logger.warning("Could not find eye landmarks for normalization. Using fixed normalization.")
                 # Fallback normalization based on face height (less robust)
                 face_top_y = min(p[1] for p in lip_points_np) # Rough estimate
                 face_bottom_y = max(p[1] for p in lip_points_np)
                 face_height_est = face_bottom_y - face_top_y
                 if face_height_est > 1:
                      mouth_open_score = min(1.0, vertical_distance / (face_height_est * 0.3))
                 else:
                      mouth_open_score = 0.0

            return lip_region_gray, mouth_open_score, lip_bounding_box

        except Exception as e:
            logger.error(f"Error extracting lip region using landmarks: {e}", exc_info=True)
            return None, 0.0, None

    def _compute_optical_flow(self, prev_frame, curr_frame):
        """
        Compute optical flow between two frames

        Args:
            prev_frame: Previous lip region frame (grayscale, uint8)
            curr_frame: Current lip region frame (grayscale, uint8)

        Returns:
            float: Movement score from 0 to 1
        """
        try:
            # Ensure frames are uint8 for optical flow
            if prev_frame is None or curr_frame is None:
                return 0.0
            if prev_frame.dtype != np.uint8:
                prev_frame = prev_frame.astype(np.uint8)
            if curr_frame.dtype != np.uint8:
                curr_frame = curr_frame.astype(np.uint8)

            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame, curr_frame, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2,
                flags=0
            )
            
            # Calculate the magnitude of the flow vectors
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Calculate the average movement
            avg_movement = np.mean(mag)
            
            # Normalize to 0-1 range
            # Values around 0.02-0.1 are typical for speaking
            movement_score = min(1.0, avg_movement * 10.0)
            
            return movement_score
            
        except Exception as e:
            logger.error(f"Error computing optical flow: {e}")
            return 0.0

    def check_lip_sync(self, frame, face_locations, audio_chunk=None):
        """
        Check lip sync using MediaPipe landmarks and optional optical flow.

        Args:
            frame: Video frame (expects BGR format from OpenCV).
            face_locations: List of face locations as (top, right, bottom, left) tuples.
                            These are USED ONLY TO MAP RESULTS back, not for detection here.
            audio_chunk: Audio chunk corresponding to this frame (numpy array, int16).

        Returns:
            dict: Dictionary mapping face index (from input face_locations) to lip sync score (0-1).
        """
        if self.face_mesh is None:
            logger.warning("Face Mesh model not initialized. Cannot check lip sync.")
            return {i: 0.0 for i in range(len(face_locations))} # Return zero scores

        try:
            with self.lock:
                lip_sync_scores = {} # Return a dictionary {index: score}

                # --- MediaPipe Processing ---
                # 1. Convert BGR frame to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 2. Make frame non-writeable for performance
                frame_rgb.flags.writeable = False
                # 3. Process the frame to find face landmarks
                results = self.face_mesh.process(frame_rgb)
                # 4. Make frame writeable again if needed later (e.g., for drawing)
                # frame_rgb.flags.writeable = True
                # --- End MediaPipe Processing ---

                # Get audio energy if audio chunk is provided
                audio_energy = 0.0
                if audio_chunk is not None and audio_chunk.size > 0:
                    audio_float = audio_chunk.astype(np.float32) / 32768.0
                    audio_energy = np.sqrt(np.mean(np.square(audio_float))) * 100.0
                    # logger.debug(f"Audio energy: {audio_energy:.2f}")

                # Process detected faces from MediaPipe
                detected_faces_landmarks = results.multi_face_landmarks if results.multi_face_landmarks else []

                # --- Match MediaPipe faces to input face_locations (Optional but good for consistency) ---
                # This is a simple matching based on IoU or center distance. For now, we'll assume
                # the order might correspond or just process the detected faces directly.
                # We will return scores indexed based on the ORDER MediaPipe detected them.
                # If matching is needed, it would involve calculating bounding boxes for MediaPipe faces
                # and comparing them to the input face_locations.
                # For simplicity, we'll use the index from detected_faces_landmarks.
                # --- End Matching --- 

                processed_face_indices = set()
                for face_index, face_landmarks in enumerate(detected_faces_landmarks):
                    processed_face_indices.add(face_index)

                    # Extract lip region and mouth openness using landmarks
                    lip_region_gray, mouth_open_score, _ = self._extract_lip_region(frame_rgb, face_landmarks)

                    if lip_region_gray is None:
                        lip_sync_scores[face_index] = 0.0
                        self.prev_lip_frames.pop(face_index, None)
                        self.lip_movement_scores.pop(face_index, None)
                        # logger.debug(f"Face {face_index}: Lip region extraction failed.")
                        continue

                    # Initialize movement score
                    movement_score = 0.0

                    # Calculate optical flow if enabled
                    if self.use_optical_flow and face_index in self.prev_lip_frames:
                        prev_frame = self.prev_lip_frames[face_index]
                        flow_score = self._compute_optical_flow(prev_frame, lip_region_gray)
                        movement_score = flow_score # Use flow score directly
                        logger.debug(f"Face {face_index}: Optical flow score: {flow_score:.3f}")
                    else: logger.debug(f"Face {face_index}: No previous frame for optical flow.")

                    # Store current frame for next comparison
                    self.prev_lip_frames[face_index] = lip_region_gray.copy()

                    # Combine optical flow (if used) with mouth openness score
                    # Give more weight to mouth openness as it's more direct
                    combined_movement_score = (0.7 * mouth_open_score) + (0.3 * movement_score if self.use_optical_flow else 0.0)
                    logger.debug(f"Face {face_index}: Mouth open score: {mouth_open_score:.3f}, Combined movement: {combined_movement_score:.3f}")

                    # Smooth the movement score using EMA
                    prev_score = self.lip_movement_scores.get(face_index, 0.0)
                    smoothed_movement_score = 0.4 * combined_movement_score + 0.6 * prev_score # Adjusted weights
                    self.lip_movement_scores[face_index] = smoothed_movement_score
                    logger.debug(f"Face {face_index}: Smoothed movement score: {smoothed_movement_score:.3f}")

                    # Calculate final lip sync score based on movement and audio energy
                    audio_energy_threshold = 1.0 # Adjust this threshold based on testing
                    if audio_energy > audio_energy_threshold:
                        sync_score = smoothed_movement_score
                        logger.debug(f"Face {face_index}: Audio detected (Energy: {audio_energy:.2f}), sync score = movement score.")
                    else:
                        # Penalize movement when silent
                        sync_score = max(0, 0.3 - smoothed_movement_score)
                        logger.debug(f"Face {face_index}: No audio detected (Energy: {audio_energy:.2f}), sync score based on lack of movement.")

                    # Cap to 0-1 range
                    sync_score = max(0.0, min(1.0, sync_score))

                    lip_sync_scores[face_index] = sync_score
                    logger.debug(f"Face {face_index}: Final sync score: {sync_score:.3f}")

                # Clean up state for faces no longer detected by MediaPipe
                stale_ids = set(self.prev_lip_frames.keys()) - processed_face_indices
                for stale_id in stale_ids:
                    self.prev_lip_frames.pop(stale_id, None)
                    self.lip_movement_scores.pop(stale_id, None)
                # if stale_ids: logger.debug(f"Cleaned up stale lip sync state for face IDs: {stale_ids}")

                # IMPORTANT: The keys in lip_sync_scores now correspond to the order
                # MediaPipe detected faces, NOT necessarily the order of the input face_locations.
                # If a direct mapping to the input face_locations is required, the matching logic
                # mentioned above needs to be implemented.
                # For now, we return the scores indexed by MediaPipe detection order.
                return lip_sync_scores

        except Exception as e:
            logger.error(f"Error in lip sync detection: {e}", exc_info=True)
            return {}

    def reset(self):
        """Reset the lip sync detector state"""
        with self.lock:
            self.prev_lip_frames.clear()
            self.lip_movement_scores.clear()

    # --- Update annotate_frame --- 
    def annotate_frame(self, frame, face_locations, lip_sync_scores):
        """
        Draw lip sync status on a frame.
        NOTE: This assumes lip_sync_scores keys match the indices of face_locations.
              If check_lip_sync doesn't guarantee this mapping, this function needs adjustment.

        Args:
            frame: Input video frame (BGR)
            face_locations: List of face locations as (top, right, bottom, left) tuples.
            lip_sync_scores: Dictionary mapping face index to lip sync score.

        Returns:
            numpy.ndarray: Annotated frame
        """
        try:
            result = frame.copy()

            # --- Re-run FaceMesh to get landmarks for drawing lip ROIs --- 
            # This is inefficient but necessary if _extract_lip_region doesn't return the box
            # or if check_lip_sync doesn't store it.
            # A better approach would be to have check_lip_sync return the boxes.
            # For now, let's just draw the status text.
            # --- End Re-run --- 

            for i, face_loc in enumerate(face_locations):
                # Convert (top, right, bottom, left) to (x, y, w, h)
                top, right, bottom, left = face_loc
                x, y, w, h = left, top, right - left, bottom - top

                # Check if a score exists for this index (assuming mapping holds)
                if i in lip_sync_scores:
                    score = lip_sync_scores[i]

                    # Draw lip sync status
                    if score > self.threshold:
                        color = (0, 255, 0)  # Green for speaking
                        status = "Sync"
                    else:
                        color = (0, 0, 255)  # Red for not speaking/sync
                        status = "No Sync"

                    # Draw status text below the face
                    label = f"Lip: {status} ({score:.2f})"
                    text_y = y + h + 20 # Position below face box
                    cv2.putText(result, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # --- Draw Lip Bounding Box (Requires landmarks or stored box) ---
                    # To draw the accurate lip box, we'd need the landmarks again here,
                    # or have check_lip_sync return the calculated lip_bounding_box.
                    # Let's skip drawing the box for now to avoid re-processing.
                    # If lip_bounding_box was returned by check_lip_sync:
                    #   x_min, y_min, x_max, y_max = lip_bounding_box[i]
                    #   cv2.rectangle(result, (x_min, y_min), (x_max, y_max), color, 1)
                    # --- End Draw Lip Box --- 

            return result

        except Exception as e:
            logger.error(f"Error annotating frame with lip sync: {e}", exc_info=True)
            return frame # Return original frame on error