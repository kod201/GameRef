#!/usr/bin/env python3
"""
Speaker Matching module for Who Dey Tallk 2

Combines multiple identification methods to determine who is speaking
"""
import logging
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)

class SpeakerMatcher:
    """
    Speaker matching decision engine
    
    Combines evidence from face recognition, lip sync detection, and
    voice biometrics to determine who is speaking.
    """
    
    def __init__(self, face_recognition_weight=0.5, lip_sync_weight=0.3, 
                 voice_biometrics_weight=0.2, min_confidence_threshold=0.65,
                 unknown_speaker_threshold=0.45):
        """
        Initialize the speaker matcher
        
        Args:
            face_recognition_weight: Weight for face recognition in decision (0.0-1.0)
            lip_sync_weight: Weight for lip sync detection in decision (0.0-1.0)
            voice_biometrics_weight: Weight for voice biometrics in decision (0.0-1.0)
            min_confidence_threshold: Minimum confidence required to identify a speaker
            unknown_speaker_threshold: Threshold below which to consider speaker unknown
        """
        self.face_recognition_weight = face_recognition_weight
        self.lip_sync_weight = lip_sync_weight
        self.voice_biometrics_weight = voice_biometrics_weight
        self.min_confidence_threshold = min_confidence_threshold
        self.unknown_speaker_threshold = unknown_speaker_threshold
        
        # Ensure weights sum to 1.0
        total_weight = face_recognition_weight + lip_sync_weight + voice_biometrics_weight
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Weights don't sum to 1.0 (sum: {total_weight:.2f}), normalizing")
            self.face_recognition_weight /= total_weight
            self.lip_sync_weight /= total_weight
            self.voice_biometrics_weight /= total_weight
        
        # Internal state
        self.lock = threading.RLock()
        self.speaker_history = []
        self.last_speaker_id = None
        self.last_speaker_confidence = 0.0
    
    def determine_speaker(self, face_identities=None, lip_sync_scores=None, voice_identity=None):
        """
        Determine who is speaking based on available evidence

        Args:
            face_identities: List of face identity dicts from face recognition
                [{'id': id, 'name': name, 'confidence': confidence}, ...]
            lip_sync_scores: Dict mapping face indices to lip sync scores
                {face_index: score, ...}  # <-- Updated type
            voice_identity: Voice identity dict from voice biometrics
                {'id': id, 'name': name, 'confidence': confidence} or None

        Returns:
            dict: {
                'speaker_id': id of the speaker or 'unknown',
                'name': name of the speaker or 'Unknown',
                'confidence': confidence score (0.0-1.0),
                'method': method used for identification ('face', 'voice', 'combined')
            } or None if no speaker detected
        """
        try:
            with self.lock:
                candidates = defaultdict(lambda: {'score': 0.0, 'name': 'Unknown', 'confidence': 0.0, 'methods': set()})
                logger.debug(f"Determining speaker. Faces: {len(face_identities) if face_identities else 0}, Lip Scores: {lip_sync_scores}, Voice: {voice_identity is not None}")

                # Process face identities with lip sync scores
                if face_identities and lip_sync_scores:
                    for i, face in enumerate(face_identities):
                        face_id = face.get('id')
                        if face_id is None:
                            logger.debug(f"Face {i}: Skipping unknown face.")
                            continue  # Skip unknown faces

                        # *** Get lip score using the index from the dictionary ***
                        lip_score = lip_sync_scores.get(i, 0.0)
                        face_confidence = face.get('confidence', 0.0)
                        logger.debug(f"Face {i} (ID: {face_id}): FaceConf={face_confidence:.2f}, LipScore={lip_score:.2f}")

                        # Only consider faces with some lip movement or high face confidence
                        # Adjust lip_score threshold as needed
                        lip_threshold_for_face = 0.2
                        if lip_score > lip_threshold_for_face or face_confidence > 0.8:
                            # Calculate weighted score contribution from face+lip
                            # Normalize weights used here if needed, or use raw weights
                            face_lip_score_contribution = (self.face_recognition_weight * face_confidence +
                                                           self.lip_sync_weight * lip_score)

                            candidates[face_id]['score'] += face_lip_score_contribution
                            candidates[face_id]['name'] = face.get('name', face_id)
                            # Store individual confidences for potential averaging later
                            candidates[face_id]['face_conf'] = face_confidence
                            candidates[face_id]['lip_score'] = lip_score
                            candidates[face_id]['methods'].add('face')
                            if lip_score > lip_threshold_for_face:
                                candidates[face_id]['methods'].add('lip')
                            logger.debug(f"Face {i} (ID: {face_id}): Added score {face_lip_score_contribution:.2f}. Total score: {candidates[face_id]['score']:.2f}")
                        else:
                             logger.debug(f"Face {i} (ID: {face_id}): Skipped due to low lip score ({lip_score:.2f}) and face confidence ({face_confidence:.2f}).")

                # Process voice identity
                if voice_identity and voice_identity.get('id') is not None:
                    voice_id = voice_identity.get('id')
                    voice_confidence = voice_identity.get('confidence', 0.0)
                    logger.debug(f"Voice ID: {voice_id}, VoiceConf={voice_confidence:.2f}")

                    # Add weighted voice score contribution
                    voice_score_contribution = self.voice_biometrics_weight * voice_confidence
                    candidates[voice_id]['score'] += voice_score_contribution
                    candidates[voice_id]['name'] = voice_identity.get('name', voice_id)
                    candidates[voice_id]['voice_conf'] = voice_confidence
                    candidates[voice_id]['methods'].add('voice')
                    logger.debug(f"Voice ID: {voice_id}: Added score {voice_score_contribution:.2f}. Total score: {candidates[voice_id]['score']:.2f}")

                # If no candidates, return None (no one identified)
                if not candidates:
                    logger.debug("No candidates found.")
                    self.last_speaker_id = None # Reset history if no candidates
                    return None

                # Calculate final confidence and find the best candidate
                best_id = None
                best_score = -1.0 # Use -1 to ensure any candidate is chosen initially
                final_candidates = {}

                for candidate_id, data in candidates.items():
                    # Calculate overall confidence (e.g., average of contributing factors)
                    conf_sum = 0.0
                    conf_count = 0
                    if 'face_conf' in data:
                        conf_sum += data['face_conf']
                        conf_count += 1
                    if 'lip_score' in data and 'lip' in data['methods']: # Only use lip score if it was above threshold
                        conf_sum += data['lip_score']
                        conf_count += 1
                    if 'voice_conf' in data:
                        conf_sum += data['voice_conf']
                        conf_count += 1

                    final_confidence = (conf_sum / conf_count) if conf_count > 0 else 0.0
                    data['confidence'] = final_confidence
                    data['method'] = "+".join(sorted(list(data['methods']))) # e.g., "face+lip", "voice", "face+lip+voice"

                    final_candidates[candidate_id] = data
                    logger.debug(f"Candidate {candidate_id} ({data['name']}): Final Score={data['score']:.2f}, Final Confidence={final_confidence:.2f}, Method={data['method']}")

                    if data['score'] > best_score:
                        best_score = data['score']
                        best_id = candidate_id

                # Check if the best candidate is confident enough
                if best_id is not None and best_score >= self.min_confidence_threshold:
                    result = {
                        'speaker_id': best_id,
                        'name': final_candidates[best_id]['name'],
                        'confidence': final_candidates[best_id]['confidence'],
                        'method': final_candidates[best_id]['method']
                    }
                    logger.debug(f"Confident match found: {result}")

                    # Update history
                    self.speaker_history.append(result)
                    while len(self.speaker_history) > 10:  # Keep last 10 results
                        self.speaker_history.pop(0)

                    self.last_speaker_id = best_id
                    self.last_speaker_confidence = final_candidates[best_id]['confidence']

                    return result

                elif self.last_speaker_id and best_score >= self.unknown_speaker_threshold:
                    # Use previous speaker if current match isn't confident enough
                    # but still above unknown threshold (persistence)
                    confidence = self.last_speaker_confidence * 0.9 # Reduce confidence slightly
                    # Try to get the name from current candidates if possible, else use ID
                    last_speaker_name = final_candidates.get(self.last_speaker_id, {}).get('name', self.last_speaker_id)

                    result = {
                        'speaker_id': self.last_speaker_id,
                        'name': last_speaker_name,
                        'confidence': confidence,
                        'method': 'history'
                    }
                    logger.debug(f"Using history for speaker: {result}")
                    return result
                else:
                    # Not confident enough, unknown speaker
                    logger.debug(f"Best score {best_score:.2f} below thresholds. Speaker unknown.")
                    self.last_speaker_id = None # Reset history if unknown
                    return {
                        'speaker_id': 'unknown',
                        'name': 'Unknown',
                        'confidence': best_score if best_score > 0 else 0.0,
                        'method': 'uncertain'
                    }

        except Exception as e:
            logger.error(f"Error determining speaker: {e}", exc_info=True)
            self.last_speaker_id = None # Reset on error
            return None

    def get_speaker_history(self):
        """
        Get the recent speaker history
        
        Returns:
            list: List of recent speaker identifications
        """
        with self.lock:
            return self.speaker_history.copy()