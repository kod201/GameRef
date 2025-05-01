#!/usr/bin/env python3
"""
Voice Biometrics module for Who Dey Tallk 2

Identifies speakers using voice embedding features
"""
import os
import logging
import numpy as np
from pathlib import Path
import sys # Added sys import

# Assume necessary imports for embedding model (e.g., torch, librosa, etc.)
# Add placeholder imports if specific libraries are unknown
try:
    import torch
    import torchaudio
    from speechbrain.pretrained import EncoderClassifier
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    # Log the specific import error
    logging.error(f"Error importing voice biometrics dependencies: {e}. Please install required packages from requirements.txt", exc_info=True)
    # Define dummy classes/functions if imports fail, allowing other parts to potentially load
    EncoderClassifier = None
    cosine_similarity = None
    torch = None
    torchaudio = None

logger = logging.getLogger(__name__)

class VoiceBiometricsEngine:
    """
    Handles voice embedding generation and speaker identification based on voice.
    """

    # Updated __init__ to remove model_path and adjust threshold default
    def __init__(self, embeddings_path, threshold=0.85):
        """
        Initialize the voice biometrics engine.

        Args:
            embeddings_path: Path to the directory containing known voice embeddings.
            threshold: Similarity threshold for identification (Cosine Similarity based).
        """
        # --- Updated Logging and Initialization ---
        logger.info("Initializing VoiceBiometricsEngine with SpeechBrain ECAPA-TDNN...")
        self.embeddings_path = Path(embeddings_path)
        self.threshold = threshold
        self.known_embeddings = {}
        self.model = None
        logger.info(f"Embeddings Path: {self.embeddings_path}")
        logger.info(f"Identification Threshold (Cosine Similarity): {self.threshold}")
        # --- End Updated Logging ---

        self._load_model() # Call the new load model
        self._load_known_embeddings() # Keep this

    # Replaced _load_model
    def _load_model(self):
        """Load the SpeechBrain ECAPA-TDNN model."""
        if EncoderClassifier is None:
             logger.error("SpeechBrain library not imported correctly. Cannot load model.")
             return

        logger.info("Attempting to load SpeechBrain ECAPA-TDNN model (may download)...")
        try:
            # This will download the model from Hugging Face Hub if not cached
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb" # Optional: specify cache dir relative to project
            )
            self.model.eval() # Set model to evaluation mode
            logger.info("Successfully loaded SpeechBrain ECAPA-TDNN model.")
        except Exception as e:
            logger.error(f"Failed to load SpeechBrain model: {e}", exc_info=True)
            self.model = None

    # Keep _load_known_embeddings as is, but ensure loaded .npy files match the new embedding size (e.g., 192)
    def _load_known_embeddings(self):
        """Load known voice embeddings from the specified directory."""
        # --- Added Logging ---
        logger.info(f"Attempting to load known voice embeddings from: {self.embeddings_path}")
        # --- End Added Logging ---
        if not self.embeddings_path.is_dir():
            logger.error(f"Embeddings directory not found or is not a directory: {self.embeddings_path}")
            return

        loaded_count = 0
        for person_dir in self.embeddings_path.iterdir():
            if person_dir.is_dir():
                person_id = person_dir.name
                name_file = person_dir / "name.txt"
                embeddings_file = person_dir / "embeddings.npy" # This file stores the *average* embedding
                count_file = person_dir / "count.txt" # Stores the number of samples averaged

                # --- Added Logging ---
                logger.debug(f"Checking directory: {person_dir}")
                # --- End Added Logging ---

                if name_file.exists() and embeddings_file.exists():
                    try:
                        with open(name_file, 'r') as f:
                            name = f.read().strip()
                        # Load the average embedding
                        embedding = np.load(embeddings_file)
                        # Load count, default to 1 if count file missing (for backward compatibility)
                        count = 1
                        if count_file.exists():
                             with open(count_file, 'r') as f:
                                  count = int(f.read().strip())

                        self.known_embeddings[person_id] = {'name': name, 'embedding': embedding, 'count': count}
                        loaded_count += 1
                        # --- Added Logging ---
                        logger.debug(f"Loaded embedding for ID: {person_id}, Name: {name}, Embedding shape: {embedding.shape}, Count: {count}")
                        if embedding.shape[-1] != 192: # ECAPA-TDNN default embedding size
                             logger.warning(f"Loaded embedding for {person_id} has unexpected shape {embedding.shape}. Expected last dimension 192.")
                        # --- End Added Logging ---
                    except Exception as e:
                        logger.error(f"Failed to load embedding for {person_id} from {person_dir}: {e}", exc_info=True)
                else:
                    # --- Added Logging ---
                    logger.warning(f"Skipping {person_dir}: Missing name.txt or embeddings.npy")
                    # --- End Added Logging ---

        # --- Added Logging ---
        logger.info(f"Finished loading known embeddings. Total loaded: {loaded_count}")
        if loaded_count == 0:
             logger.warning("No known voice embeddings were loaded. Identification will not be possible.")
        # --- End Added Logging ---

    # Replaced _generate_embedding
    def _generate_embedding(self, audio_chunk_np, sample_rate=16000):
        """Generate a voice embedding using the SpeechBrain model."""
        if self.model is None or torch is None or torchaudio is None:
            logger.error("SpeechBrain model or dependencies not loaded correctly. Cannot generate embedding.")
            return None
        try:
            # Ensure audio is float32 tensor and correct sample rate (model expects 16kHz)
            audio_tensor = torch.tensor(audio_chunk_np.astype(np.float32))
            if audio_tensor.ndim > 1: # Ensure mono
                 audio_tensor = torch.mean(audio_tensor, dim=1)

            # Resample if necessary (model expects 16kHz)
            if sample_rate != 16000:
                # Check if torchaudio.transforms is available
                if hasattr(torchaudio, 'transforms') and hasattr(torchaudio.transforms, 'Resample'):
                    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                    audio_tensor = resampler(audio_tensor)
                else:
                    logger.warning(f"torchaudio.transforms.Resample not available. Cannot resample audio from {sample_rate}Hz to 16000Hz. Ensure input audio is 16kHz.")
                    if sample_rate != 16000:
                         return None # Cannot proceed if resampling needed but unavailable

            # Add batch dimension and move to appropriate device (CPU/GPU)
            # Check if CUDA is available and model is on GPU, otherwise use CPU
            device = next(self.model.parameters()).device
            audio_tensor = audio_tensor.unsqueeze(0).to(device)

            # Generate embedding
            with torch.no_grad():
                # The encode_batch method returns embeddings
                embedding = self.model.encode_batch(audio_tensor)
                embedding = embedding.squeeze().cpu().numpy() # Remove batch dim, move to CPU, convert to numpy

            logger.debug(f"Generated SpeechBrain embedding shape: {embedding.shape}") # Shape is likely (192,)
            return embedding

        except Exception as e:
            logger.error(f"Error generating SpeechBrain embedding: {e}", exc_info=True)
            return None

    # Replaced _calculate_similarity
    def _calculate_similarity(self, emb1, emb2):
        """Calculate cosine similarity between two embeddings."""
        if emb1 is None or emb2 is None or cosine_similarity is None:
            logger.warning("Cannot calculate similarity: Embeddings missing or cosine_similarity not available.")
            return 0.0
        try:
            # Ensure embeddings are 2D arrays for cosine_similarity
            emb1 = emb1.reshape(1, -1)
            emb2 = emb2.reshape(1, -1)
            similarity = cosine_similarity(emb1, emb2)[0][0]
            logger.debug(f"Calculated cosine similarity: {similarity:.4f}")
            # Clamp similarity to [0, 1] just in case of numerical issues
            return max(0.0, min(1.0, float(similarity)))
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}", exc_info=True)
            return 0.0

    # Updated identify_speaker to use the average embedding loaded
    def identify_speaker(self, audio_chunk_np, sample_rate=16000):
        """
        Identify the speaker from an audio chunk using average embeddings.

        Args:
            audio_chunk_np: NumPy array containing the audio data.
            sample_rate: Sample rate of the audio_chunk_np.

        Returns:
            dict: {'id': speaker_id, 'name': speaker_name, 'confidence': score}
                  if a speaker is identified above the threshold, otherwise None.
        """
        if self.model is None:
            logger.warning("Voice model not loaded. Cannot identify speaker.")
            return None

        if not self.known_embeddings:
            logger.warning("No known voice embeddings for identification.")
            return None

        # 1. Generate embedding for the input chunk, passing sample rate
        current_embedding = self._generate_embedding(audio_chunk_np, sample_rate=sample_rate)
        if current_embedding is None:
            logger.error("Failed to generate embedding for the current audio chunk.")
            return None

        # 2. Compare with known average embeddings
        best_match_id = None
        best_match_name = None
        best_similarity = -1.0

        for person_id, data in self.known_embeddings.items():
            # Use the stored average embedding for comparison
            known_avg_embedding = data['embedding']
            name = data['name']

            # Ensure embeddings are compatible for comparison
            if current_embedding.shape != known_avg_embedding.shape:
                 if current_embedding.size == known_avg_embedding.size and current_embedding.size == 192:
                      logger.debug(f"Embeddings for {person_id} have compatible size but different shape ({current_embedding.shape} vs {known_avg_embedding.shape}). Reshaping for comparison.")
                 else:
                      logger.warning(f"Shape mismatch between current ({current_embedding.shape}) and known average ({known_avg_embedding.shape}) embedding for {person_id}. Expected size 192. Skipping.")
                      continue

            similarity = self._calculate_similarity(current_embedding, known_avg_embedding)
            logger.debug(f"Comparing with {name} (ID: {person_id}): Similarity = {similarity:.4f}")

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = person_id
                best_match_name = name

        # 3. Check threshold and return result
        if best_match_id is not None and best_similarity >= self.threshold:
            result = {
                'id': best_match_id,
                'name': best_match_name,
                'confidence': best_similarity
            }
            logger.info(f"Voice match found: {best_match_name} (ID: {best_match_id}) with confidence {best_similarity:.4f}")
            return result
        else:
            logger.info(f"No voice match found above threshold ({self.threshold:.2f}). Best score: {best_similarity:.4f}")
            return None

    # Updated add_voice_embedding to call the modified save logic
    def add_voice_embedding(self, person_id, name, audio_data_np, sample_rate=16000):
        """
        Add or enrich a voice embedding using averaging.

        Args:
            person_id: Unique identifier for the person.
            name: The name of the person.
            audio_data_np: NumPy array containing the audio data for the person's voice.
            sample_rate: Sample rate of the audio_data_np.

        Returns:
            bool: True if the embedding was successfully processed and saved/updated, False otherwise.
        """
        action = "Enriching" if person_id in self.known_embeddings else "Adding"
        logger.info(f"{action} voice embedding for {name} (ID: {person_id})")
        new_embedding = self._generate_embedding(audio_data_np, sample_rate=sample_rate)

        if new_embedding is not None:
            # Check shape before saving
            if new_embedding.shape[-1] != 192:
                 logger.error(f"Generated embedding for {name} has unexpected shape {new_embedding.shape}. Cannot {action.lower()}.")

            # Save/Update the embedding using the averaging logic
            save_success = self._save_embedding(person_id, name, new_embedding)
            return save_success # Return success status from save operation
        else:
            logger.error(f"Failed to generate embedding for {name} (ID: {person_id}). Cannot {action.lower()}.")
            return False # Indicate failure

    # Modified _save_embedding to implement averaging logic
    def _save_embedding(self, person_id, name, new_embedding):
        """
        Save/update the voice embedding for a person using averaging.

        Args:
            person_id: Unique identifier for the person.
            name: The name of the person.
            new_embedding: The newly generated embedding (NumPy array) to add/average.

        Returns:
            bool: True if saving was successful, False otherwise.
        """
        person_dir = self.embeddings_path / person_id
        person_dir.mkdir(exist_ok=True)

        name_file = person_dir / "name.txt"
        embeddings_file = person_dir / "embeddings.npy" # Stores the average
        count_file = person_dir / "count.txt" # Stores the count

        old_avg_embedding = np.zeros_like(new_embedding)
        old_count = 0

        # Try to load existing average embedding and count
        if embeddings_file.exists() and count_file.exists():
            try:
                old_avg_embedding = np.load(embeddings_file)
                with open(count_file, 'r') as f:
                    old_count = int(f.read().strip())
                # --- Shape validation ---
                if old_avg_embedding.shape != new_embedding.shape:
                     logger.warning(f"Existing average embedding shape {old_avg_embedding.shape} differs from new embedding shape {new_embedding.shape} for {person_id}. Resetting average.")
                     old_avg_embedding = np.zeros_like(new_embedding)
                     old_count = 0
                # --- End shape validation ---
            except Exception as e:
                logger.error(f"Failed to load existing embedding/count for {person_id}, will overwrite: {e}", exc_info=True)
                old_avg_embedding = np.zeros_like(new_embedding)
                old_count = 0
        elif embeddings_file.exists() != count_file.exists():
             logger.warning(f"Inconsistent state for {person_id}: Found one of embeddings.npy/count.txt but not both. Will overwrite.")
             old_avg_embedding = np.zeros_like(new_embedding)
             old_count = 0


        # Calculate new average embedding
        new_count = old_count + 1
        # Formula: new_avg = (old_avg * old_count + new_emb) / new_count
        new_avg_embedding = ((old_avg_embedding * old_count) + new_embedding) / new_count

        # Save the new average embedding, name, and count
        try:
            with open(name_file, 'w') as f:
                f.write(name)
            np.save(embeddings_file, new_avg_embedding)
            with open(count_file, 'w') as f:
                f.write(str(new_count))

            # Update in-memory store
            self.known_embeddings[person_id] = {'name': name, 'embedding': new_avg_embedding, 'count': new_count}
            logger.info(f"Saved/Updated average embedding for {name} (ID: {person_id}). New count: {new_count}")
            return True # Indicate success
        except Exception as e:
            logger.error(f"Failed to save updated embedding for {person_id}: {e}", exc_info=True)
            return False # Indicate failure