#!/usr/bin/env python3
"""
Database Manager Module for Who Dey Tallk 2

Handles storage and retrieval of conversation data
"""
import os
import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Database manager for Who Dey Tallk 2

    Handles storage and retrieval of:
    - Transcription data
    - Speaker identities
    - Conversation history
    """

    def __init__(self, config): # Accept config object
        """
        Initialize the database manager

        Args:
            config: Configuration dictionary containing database settings
        """
        self.config = config # Store the whole config if needed elsewhere
        self.db_config = config.get("database", {})

        # Get database path from config or use default
        self.base_dir = Path(__file__).parent.parent # Project root
        db_path_str = self.db_config.get("path", "database/conversations.db") # Use conversations.db as default
        self.db_path = self.base_dir / db_path_str

        # Ensure database directory exists
        os.makedirs(self.db_path.parent, exist_ok=True)

        # Initialize database connection
        self.conn = None
        self.cursor = None

        # Connect to database
        self._connect()

        # Create database schema if needed
        self._initialize_schema()

    def _connect(self):
        """Connect to the SQLite database"""
        try:
            self.conn = sqlite3.connect(
                str(self.db_path),
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
                check_same_thread=False # Allow connection sharing across threads
            )

            # Enable foreign keys
            self.conn.execute("PRAGMA foreign_keys = ON")

            # Set row factory
            self.conn.row_factory = sqlite3.Row

            # Create cursor
            self.cursor = self.conn.cursor()

            logger.info(f"Connected to database: {self.db_path}")

        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            raise

    def _disconnect(self):
        """Disconnect from the database"""
        if self.conn:
            try:
                self.conn.close()
                logger.debug("Disconnected from database")
            except sqlite3.Error as e:
                logger.error(f"Error disconnecting from database: {e}")

    def _initialize_schema(self):
        """Initialize database schema if not exists"""
        try:
            # Create speakers table with external_id
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS speakers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    external_id TEXT UNIQUE, -- Added for linking recognizer ID
                    name TEXT NOT NULL,
                    embedding BLOB,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    recognition_count INTEGER DEFAULT 0,
                    metadata TEXT
                )
            """)
            # --- BEGIN ADDED CODE ---
            # Check if 'external_id' column exists and add it if missing (schema migration)
            self.cursor.execute("PRAGMA table_info(speakers)")
            columns = [column[1] for column in self.cursor.fetchall()]
            if 'external_id' not in columns:
                logger.info("Attempting to add missing 'external_id' column to 'speakers' table.")
                try:
                    # Add the column. Note: Adding UNIQUE constraint might fail if existing rows have NULLs/duplicates.
                    # Consider a more robust migration strategy for production.
                    # Add the column *without* the UNIQUE constraint first
                    self.cursor.execute("ALTER TABLE speakers ADD COLUMN external_id TEXT")
                    self.conn.commit() # Commit the schema change immediately
                    logger.info("Successfully added 'external_id' column.")
                except sqlite3.Error as alter_err:
                    logger.error(f"Failed to add 'external_id' column: {alter_err}. Manual schema update might be required.")
                    self.conn.rollback() # Rollback if ALTER fails
                    raise # Re-raise the error as schema update failed
            # --- END ADDED CODE ---

            # Add UNIQUE index for faster lookup and uniqueness enforcement (allows multiple NULLs)
            self.cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_speakers_external_id ON speakers (external_id)")

            # Create conversations table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- Ensure this column exists
                    end_time TIMESTAMP,
                    title TEXT,
                    summary TEXT,
                    metadata TEXT
                )
            """)
            # --- BEGIN ADDED CODE for conversations migration ---
            # Check if 'start_time' column exists in conversations and add it if missing
            self.cursor.execute("PRAGMA table_info(conversations)")
            convo_columns = [column[1] for column in self.cursor.fetchall()]
            if 'start_time' not in convo_columns:
                logger.info("Attempting to add missing 'start_time' column to 'conversations' table.")
                try:
                    # Add the column *without* the default constraint first
                    self.cursor.execute("ALTER TABLE conversations ADD COLUMN start_time TIMESTAMP")
                    # Optionally, update existing rows if a default is needed for them
                    # self.cursor.execute("UPDATE conversations SET start_time = ? WHERE start_time IS NULL", (datetime.now(),))
                    self.conn.commit() # Commit the schema change immediately
                    logger.info("Successfully added 'start_time' column to 'conversations'. Existing rows will have NULL start_time initially.")
                except sqlite3.Error as alter_err:
                    logger.error(f"Failed to add 'start_time' column to 'conversations': {alter_err}. Manual schema update might be required.")
                    self.conn.rollback() # Rollback if ALTER fails
                    raise # Re-raise the error as schema update failed
            # --- END ADDED CODE for conversations migration ---

            # --- BEGIN ADDED CODE for conversations title migration ---
            # Check if 'title' column exists in conversations and add it if missing
            self.cursor.execute("PRAGMA table_info(conversations)")
            convo_columns_for_title = [column[1] for column in self.cursor.fetchall()]
            if 'title' not in convo_columns_for_title:
                logger.info("Attempting to add missing 'title' column to 'conversations' table.")
                try:
                    # Add the column
                    self.cursor.execute("ALTER TABLE conversations ADD COLUMN title TEXT")
                    self.conn.commit() # Commit the schema change immediately
                    logger.info("Successfully added 'title' column to 'conversations'.")
                except sqlite3.Error as alter_err:
                    logger.error(f"Failed to add 'title' column to 'conversations': {alter_err}. Manual schema update might be required.")
                    self.conn.rollback() # Rollback if ALTER fails
                    raise # Re-raise the error as schema update failed
            # --- END ADDED CODE for conversations title migration ---

            # --- BEGIN ADDED CODE for conversations metadata migration ---
            # Check if 'metadata' column exists in conversations and add it if missing
            self.cursor.execute("PRAGMA table_info(conversations)")
            convo_columns_for_metadata = [column[1] for column in self.cursor.fetchall()]
            if 'metadata' not in convo_columns_for_metadata:
                logger.info("Attempting to add missing 'metadata' column to 'conversations' table.")
                try:
                    # Add the column
                    self.cursor.execute("ALTER TABLE conversations ADD COLUMN metadata TEXT")
                    self.conn.commit() # Commit the schema change immediately
                    logger.info("Successfully added 'metadata' column to 'conversations'.")
                except sqlite3.Error as alter_err:
                    logger.error(f"Failed to add 'metadata' column to 'conversations': {alter_err}. Manual schema update might be required.")
                    self.conn.rollback() # Rollback if ALTER fails
                    raise # Re-raise the error as schema update failed
            # --- END ADDED CODE for conversations metadata migration ---

            # --- BEGIN ADDED CODE to remove incorrect speaker_id from conversations ---
            self.cursor.execute("PRAGMA table_info(conversations)")
            convo_columns_final_check = {column[1]: column for column in self.cursor.fetchall()}
            if 'speaker_id' in convo_columns_final_check:
                logger.warning("Detected unexpected 'speaker_id' column in 'conversations' table. Attempting to remove it.")
                try:
                    # Attempt to drop the column (Requires SQLite 3.35.0+)
                    self.cursor.execute("ALTER TABLE conversations DROP COLUMN speaker_id")
                    self.conn.commit()
                    logger.info("Successfully removed 'speaker_id' column from 'conversations' table.")
                except sqlite3.Error as drop_err:
                    # Older SQLite versions might not support DROP COLUMN
                    logger.error(f"Failed to remove 'speaker_id' column from 'conversations': {drop_err}. This might be due to an older SQLite version. Manual schema correction (e.g., recreating the table) might be required.")
                    self.conn.rollback()
                    # Decide whether to raise the error or continue with the potentially incorrect schema
                    # raise # Option 1: Stop execution if schema correction fails
                    # Option 2: Log the error and continue (might lead to issues later)
                    pass # For now, log and continue, but be aware of potential issues
            # --- END ADDED CODE to remove incorrect speaker_id from conversations ---

            # Create utterances table (linking to speakers.id)
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS utterances (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL, -- Changed: Ensure it's not null
                    speaker_id INTEGER, -- Internal DB ID, can be NULL for unknown
                    text TEXT NOT NULL,
                    confidence REAL DEFAULT 0.0,
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    duration REAL DEFAULT 0.0,
                    audio_path TEXT,
                    metadata TEXT,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE, -- Added ON DELETE CASCADE
                    FOREIGN KEY(speaker_id) REFERENCES speakers(id) ON DELETE SET NULL -- Added ON DELETE SET NULL
                )
            """)
            # Add index for faster lookup
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_utterances_conversation_id ON utterances (conversation_id)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_utterances_speaker_id ON utterances (speaker_id)")


            # Create settings table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create voiceprints table (linking to speakers.id)
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS voiceprints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    speaker_id INTEGER NOT NULL, -- Internal DB ID
                    embedding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source_audio_path TEXT,
                    metadata TEXT,
                    FOREIGN KEY(speaker_id) REFERENCES speakers(id) ON DELETE CASCADE -- Added ON DELETE CASCADE
                )
            """)
            # Add index for faster lookup
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_voiceprints_speaker_id ON voiceprints (speaker_id)")


            # Commit changes
            self.conn.commit()
            logger.info("Database schema initialized")

        except sqlite3.Error as e:
            logger.error(f"Error initializing database schema: {e}")
            self.conn.rollback()
            raise

    def close(self):
        """Close database connection"""
        self._disconnect()

    def start_conversation(self, title=None, metadata=None):
        """
        Start a new conversation

        Args:
            title: Optional conversation title
            metadata: Optional metadata dictionary

        Returns:
            int: ID of the new conversation
        """
        try:
            # Convert metadata to JSON if provided
            metadata_json = None
            if metadata:
                metadata_json = json.dumps(metadata)

            # Insert new conversation
            self.cursor.execute(
                """
                INSERT INTO conversations (start_time, title, metadata)
                VALUES (CURRENT_TIMESTAMP, ?, ?)
                """,
                (title, metadata_json)
            )

            # Get the conversation ID
            conversation_id = self.cursor.lastrowid

            # Commit changes
            self.conn.commit()

            logger.info(f"Started new conversation: {conversation_id}")
            return conversation_id

        except sqlite3.Error as e:
            logger.error(f"Error starting conversation: {e}")
            self.conn.rollback()
            return None

    def end_conversation(self, conversation_id, summary=None):
        """
        End an existing conversation

        Args:
            conversation_id: ID of the conversation to end
            summary: Optional conversation summary

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Update conversation with end time and summary
            self.cursor.execute(
                """
                UPDATE conversations
                SET end_time = CURRENT_TIMESTAMP, summary = ?
                WHERE id = ?
                """,
                (summary, conversation_id)
            )

            # Commit changes
            self.conn.commit()

            logger.info(f"Ended conversation: {conversation_id}")
            return True

        except sqlite3.Error as e:
            logger.error(f"Error ending conversation: {e}")
            self.conn.rollback()
            return False

    def add_utterance(self, conversation_id, text, speaker_external_id=None, confidence=0.0,
                     duration=0.0, audio_path=None, metadata=None):
        """
        Add an utterance to a conversation, looking up speaker by external_id.

        Args:
            conversation_id: ID of the conversation
            text: Transcribed text
            speaker_external_id: Optional speaker external ID (string like 'stephen' or 'unknown')
            confidence: Transcription confidence
            duration: Audio duration in seconds
            audio_path: Optional path to audio file
            metadata: Optional metadata dictionary

        Returns:
            int: ID of the new utterance or None on error
        """
        try:
            speaker_db_id = None
            if speaker_external_id and speaker_external_id.lower() != 'unknown':
                speaker_db_id = self.get_speaker_db_id(speaker_external_id)
                if speaker_db_id is None:
                    logger.warning(f"Speaker with external_id '{speaker_external_id}' not found in DB. Storing utterance as unknown.")

            # Convert metadata to JSON if provided
            metadata_json = json.dumps(metadata) if metadata else None

            # Insert new utterance
            self.cursor.execute(
                """
                INSERT INTO utterances (
                    conversation_id, speaker_id, text, confidence,
                    start_time, duration, audio_path, metadata
                )
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?)
                """,
                (conversation_id, speaker_db_id, text, confidence,
                 duration, audio_path, metadata_json)
            )

            # Get the utterance ID
            utterance_id = self.cursor.lastrowid

            # Update speaker last_seen if speaker was identified
            if speaker_db_id:
                self.cursor.execute(
                    """
                    UPDATE speakers
                    SET last_seen = CURRENT_TIMESTAMP,
                        recognition_count = recognition_count + 1
                    WHERE id = ?
                    """,
                    (speaker_db_id,)
                )

            # Commit changes
            self.conn.commit()

            logger.debug(f"Added utterance {utterance_id} to conversation {conversation_id} (Speaker DB ID: {speaker_db_id})")
            return utterance_id

        except sqlite3.Error as e:
            logger.error(f"Error adding utterance: {e}", exc_info=True)
            self.conn.rollback()
            return None

    def add_or_update_speaker(self, external_id, name, embedding=None, metadata=None):
        """
        Add a new speaker or update the name of an existing one based on external_id.

        Args:
            external_id: The unique external identifier (e.g., from --add-face)
            name: Speaker name
            embedding: Optional speaker embedding (voice print) - Note: This might be better handled via add_voiceprint
            metadata: Optional metadata dictionary

        Returns:
            int: Database ID (primary key) of the added/updated speaker or None on error
        """
        if not external_id:
            logger.error("Cannot add or update speaker without an external_id.")
            return None

        try:
            metadata_json = json.dumps(metadata) if metadata else None
            embedding_blob = sqlite3.Binary(embedding) if embedding else None

            # Try to insert, or update name if external_id constraint fails
            self.cursor.execute("""
                INSERT INTO speakers (external_id, name, embedding, metadata)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(external_id) DO UPDATE SET
                    name = excluded.name,
                    last_seen = CURRENT_TIMESTAMP -- Update last_seen on update too
                    -- Decide if embedding/metadata should be updated here or only on insert
                    -- metadata = excluded.metadata -- Example: update metadata if needed
            """, (external_id, name, embedding_blob, metadata_json))

            self.conn.commit()

            # Get the ID of the inserted/updated row
            speaker_db_id = self.get_speaker_db_id(external_id)

            if speaker_db_id:
                 logger.info(f"Added/Updated speaker: {name} (External ID: {external_id}, DB ID: {speaker_db_id})")
            else:
                 # This case should ideally not happen if insert/update worked and get_speaker_db_id is correct
                 logger.error(f"Failed to retrieve DB ID after adding/updating speaker with external_id: {external_id}")


            return speaker_db_id

        except sqlite3.Error as e:
            logger.error(f"Error adding/updating speaker (External ID: {external_id}): {e}", exc_info=True)
            self.conn.rollback()
            return None

    def get_speaker_db_id(self, external_id):
        """
        Get the internal database ID for a given external speaker ID.

        Args:
            external_id: The external identifier string.

        Returns:
            int: The internal database ID (primary key) or None if not found.
        """
        if not external_id:
            return None
        try:
            self.cursor.execute(
                "SELECT id FROM speakers WHERE external_id = ?",
                (external_id,)
            )
            result = self.cursor.fetchone()
            return result['id'] if result else None
        except sqlite3.Error as e:
            logger.error(f"Error getting speaker DB ID for external_id '{external_id}': {e}")
            return None

    # Remove the old add_speaker method if it's fully replaced by add_or_update_speaker
    # def add_speaker(self, name, embedding=None, metadata=None): ...

    # ... rest of the methods (get_speaker, get_conversation, etc.) ...
    # Note: get_speaker should probably fetch by internal ID,
    # maybe add get_speaker_by_external_id if needed.

    def get_speaker(self, speaker_db_id): # Changed parameter name for clarity
        """
        Get speaker by internal database ID
        Args:
            speaker_db_id: Internal ID (primary key) of the speaker
        Returns:
            dict: Speaker data or None if not found
        """
        try:
            self.cursor.execute(
                """
                SELECT id, external_id, name, embedding, first_seen, last_seen,
                       recognition_count, metadata
                FROM speakers
                WHERE id = ?
                """,
                (speaker_db_id,)
            )
            row = self.cursor.fetchone()
            if not row:
                # logger.debug(f"Speaker not found with DB ID: {speaker_db_id}") # Can be noisy
                return None
            speaker = dict(row)
            if speaker["metadata"]:
                speaker["metadata"] = json.loads(speaker["metadata"])
            return speaker
        except sqlite3.Error as e:
            logger.error(f"Error getting speaker by DB ID {speaker_db_id}: {e}")
            return None

    def get_conversation(self, conversation_id):
        """
        Get conversation by ID

        Args:
            conversation_id: ID of the conversation

        Returns:
            dict: Conversation data or None if not found
        """
        try:
            # Query conversation
            self.cursor.execute(
                """
                SELECT id, start_time, end_time, title, summary, metadata
                FROM conversations
                WHERE id = ?
                """,
                (conversation_id,)
            )

            # Get result
            row = self.cursor.fetchone()

            if not row:
                logger.debug(f"Conversation not found: {conversation_id}")
                return None

            # Convert to dictionary
            conversation = dict(row)

            # Parse metadata JSON if exists
            if conversation["metadata"]:
                conversation["metadata"] = json.loads(conversation["metadata"])

            return conversation

        except sqlite3.Error as e:
            logger.error(f"Error getting conversation: {e}")
            return None

    def get_conversation_utterances(self, conversation_id):
        """
        Get all utterances for a conversation

        Args:
            conversation_id: ID of the conversation

        Returns:
            list: List of utterance dictionaries
        """
        try:
            # Query utterances
            self.cursor.execute(
                """
                SELECT u.id, u.conversation_id, u.speaker_id, u.text,
                       u.confidence, u.start_time, u.duration,
                       u.audio_path, u.metadata,
                       s.name as speaker_name
                FROM utterances u
                LEFT JOIN speakers s ON u.speaker_id = s.id
                WHERE u.conversation_id = ?
                ORDER BY u.start_time ASC
                """,
                (conversation_id,)
            )

            # Get results
            rows = self.cursor.fetchall()

            # Convert to list of dictionaries
            utterances = []

            for row in rows:
                utterance = dict(row)

                # Parse metadata JSON if exists
                if utterance["metadata"]:
                    utterance["metadata"] = json.loads(utterance["metadata"])

                utterances.append(utterance)

            return utterances

        except sqlite3.Error as e:
            logger.error(f"Error getting conversation utterances: {e}")
            return []

    def get_recent_conversations(self, limit=10):
        """
        Get recent conversations

        Args:
            limit: Maximum number of conversations to return

        Returns:
            list: List of conversation dictionaries
        """
        try:
            # Query conversations
            self.cursor.execute(
                """
                SELECT id, start_time, end_time, title, summary, metadata
                FROM conversations
                ORDER BY start_time DESC
                LIMIT ?
                """,
                (limit,)
            )

            # Get results
            rows = self.cursor.fetchall()

            # Convert to list of dictionaries
            conversations = []

            for row in rows:
                conversation = dict(row)

                # Parse metadata JSON if exists
                if conversation["metadata"]:
                    conversation["metadata"] = json.loads(conversation["metadata"])

                conversations.append(conversation)

            return conversations

        except sqlite3.Error as e:
            logger.error(f"Error getting recent conversations: {e}")
            return []

    def get_all_speakers(self):
        """
        Get all speakers

        Returns:
            list: List of speaker dictionaries
        """
        try:
            # Query speakers
            self.cursor.execute(
                """
                SELECT id, external_id, name, embedding, first_seen, last_seen,
                       recognition_count, metadata
                FROM speakers
                ORDER BY name ASC
                """
            )

            # Get results
            rows = self.cursor.fetchall()

            # Convert to list of dictionaries
            speakers = []

            for row in rows:
                speaker = dict(row)

                # Parse metadata JSON if exists
                if speaker["metadata"]:
                    speaker["metadata"] = json.loads(speaker["metadata"])

                speakers.append(speaker)

            return speakers

        except sqlite3.Error as e:
            logger.error(f"Error getting all speakers: {e}")
            return []

    def search_utterances(self, search_text, limit=50):
        """
        Search utterances for text

        Args:
            search_text: Text to search for
            limit: Maximum number of results to return

        Returns:
            list: List of utterance dictionaries
        """
        try:
            # Prepare search query
            search_pattern = f"%{search_text}%"

            # Query utterances
            self.cursor.execute(
                """
                SELECT u.id, u.conversation_id, u.speaker_id, u.text,
                       u.confidence, u.start_time, u.duration,
                       u.audio_path, u.metadata,
                       s.name as speaker_name,
                       c.title as conversation_title
                FROM utterances u
                LEFT JOIN speakers s ON u.speaker_id = s.id
                LEFT JOIN conversations c ON u.conversation_id = c.id
                WHERE u.text LIKE ?
                ORDER BY u.start_time DESC
                LIMIT ?
                """,
                (search_pattern, limit)
            )

            # Get results
            rows = self.cursor.fetchall()

            # Convert to list of dictionaries
            utterances = []

            for row in rows:
                utterance = dict(row)

                # Parse metadata JSON if exists
                if utterance["metadata"]:
                    utterance["metadata"] = json.loads(utterance["metadata"])

                utterances.append(utterance)

            return utterances

        except sqlite3.Error as e:
            logger.error(f"Error searching utterances: {e}")
            return []

    def get_setting(self, key, default=None):
        """
        Get a setting value

        Args:
            key: Setting key
            default: Default value if setting not found

        Returns:
            str: Setting value or default
        """
        try:
            # Query setting
            self.cursor.execute(
                """
                SELECT value
                FROM settings
                WHERE key = ?
                """,
                (key,)
            )

            # Get result
            row = self.cursor.fetchone()

            if not row:
                return default

            return row["value"]

        except sqlite3.Error as e:
            logger.error(f"Error getting setting: {e}")
            return default

    def set_setting(self, key, value):
        """
        Set a setting value

        Args:
            key: Setting key
            value: Setting value

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Update or insert setting
            self.cursor.execute(
                """
                INSERT OR REPLACE INTO settings (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                """,
                (key, value)
            )

            # Commit changes
            self.conn.commit()

            logger.debug(f"Set setting {key} = {value}")
            return True

        except sqlite3.Error as e:
            logger.error(f"Error setting setting: {e}")
            self.conn.rollback()
            return False

    def delete_conversation(self, conversation_id):
        """
        Delete a conversation and its utterances

        Args:
            conversation_id: ID of the conversation

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Begin transaction
            self.conn.execute("BEGIN")

            # Delete utterances
            self.cursor.execute(
                """
                DELETE FROM utterances
                WHERE conversation_id = ?
                """,
                (conversation_id,)
            )

            # Delete conversation
            self.cursor.execute(
                """
                DELETE FROM conversations
                WHERE id = ?
                """,
                (conversation_id,)
            )

            # Commit changes
            self.conn.commit()

            logger.info(f"Deleted conversation: {conversation_id}")
            return True

        except sqlite3.Error as e:
            logger.error(f"Error deleting conversation: {e}")
            self.conn.rollback()
            return False

    def delete_speaker(self, speaker_db_id): # Changed parameter name for clarity
        """
        Delete a speaker and their voiceprints by internal DB ID.
        Utterances linked to this speaker will have speaker_id set to NULL.

        Args:
            speaker_db_id: Internal ID (primary key) of the speaker

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Begin transaction (already handles foreign key constraints based on schema)
            # Foreign key constraint `ON DELETE SET NULL` handles utterances
            # Foreign key constraint `ON DELETE CASCADE` handles voiceprints

            # Delete speaker (triggers cascades/set null)
            self.cursor.execute(
                "DELETE FROM speakers WHERE id = ?",
                (speaker_db_id,)
            )
            rows_affected = self.cursor.rowcount

            self.conn.commit()

            if rows_affected > 0:
                logger.info(f"Deleted speaker with DB ID: {speaker_db_id}")
                return True
            else:
                logger.warning(f"Attempted to delete speaker with DB ID {speaker_db_id}, but speaker was not found.")
                return False

        except sqlite3.Error as e:
            logger.error(f"Error deleting speaker with DB ID {speaker_db_id}: {e}", exc_info=True)
            self.conn.rollback()
            return False