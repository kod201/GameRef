#!/usr/bin/env python3
"""
Database Manager for Who Dey Tallk 2

Handles storage and retrieval of conversation data, speaker profiles,
and system metrics.
"""
import os
import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database manager for Who Dey Tallk system"""
    
    # SQL statements for creating tables
    CREATE_TABLES = [
        """
        CREATE TABLE IF NOT EXISTS speakers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            face_encoding BLOB,
            voice_profile BLOB,
            notes TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            end_time TIMESTAMP,
            title TEXT,
            location TEXT,
            context TEXT,
            metadata TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS utterances (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            speaker_id INTEGER,
            speaker_name TEXT,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            text TEXT,
            confidence REAL,
            audio_path TEXT,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id),
            FOREIGN KEY (speaker_id) REFERENCES speakers(id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metric_name TEXT,
            metric_value REAL,
            context TEXT
        )
        """
    ]
    
    def __init__(self, db_path="database/conversations.db"):
        """
        Initialize the database manager
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database connection
        self.conn = None
        self.initialize_db()
    
    def initialize_db(self):
        """Initialize database connection and create tables if needed"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
            
            cursor = self.conn.cursor()
            
            # Create tables
            for create_table_sql in self.CREATE_TABLES:
                cursor.execute(create_table_sql)
            
            self.conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
            
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            if self.conn:
                self.conn.close()
                self.conn = None
            raise
    
    def ensure_connection(self):
        """Ensure database connection is active"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def add_speaker(self, name, face_encoding=None, voice_profile=None, notes=None):
        """
        Add a new speaker to the database
        
        Args:
            name: Speaker name
            face_encoding: Binary face encoding data
            voice_profile: Binary voice profile data
            notes: Additional notes about the speaker
            
        Returns:
            int: ID of the added speaker
        """
        try:
            self.ensure_connection()
            cursor = self.conn.cursor()
            
            cursor.execute(
                """
                INSERT INTO speakers (name, face_encoding, voice_profile, notes)
                VALUES (?, ?, ?, ?)
                """,
                (name, face_encoding, voice_profile, notes)
            )
            
            self.conn.commit()
            speaker_id = cursor.lastrowid
            logger.info(f"Added new speaker '{name}' with ID {speaker_id}")
            return speaker_id
            
        except sqlite3.Error as e:
            logger.error(f"Error adding speaker: {e}")
            return None
    
    def get_speaker(self, speaker_id=None, name=None):
        """
        Get speaker information by ID or name
        
        Args:
            speaker_id: Speaker ID (optional)
            name: Speaker name (optional)
            
        Returns:
            dict: Speaker information or None if not found
        """
        try:
            self.ensure_connection()
            cursor = self.conn.cursor()
            
            if speaker_id is not None:
                cursor.execute("SELECT * FROM speakers WHERE id = ?", (speaker_id,))
            elif name is not None:
                cursor.execute("SELECT * FROM speakers WHERE name = ?", (name,))
            else:
                logger.error("Either speaker_id or name must be provided")
                return None
                
            row = cursor.fetchone()
            return dict(row) if row else None
            
        except sqlite3.Error as e:
            logger.error(f"Error getting speaker: {e}")
            return None
    
    def get_all_speakers(self):
        """
        Get all speakers from the database
        
        Returns:
            list: List of speaker dictionaries
        """
        try:
            self.ensure_connection()
            cursor = self.conn.cursor()
            
            cursor.execute("SELECT * FROM speakers ORDER BY name")
            return [dict(row) for row in cursor.fetchall()]
            
        except sqlite3.Error as e:
            logger.error(f"Error getting speakers: {e}")
            return []
    
    def update_speaker(self, speaker_id, name=None, face_encoding=None, 
                      voice_profile=None, notes=None):
        """
        Update speaker information
        
        Args:
            speaker_id: ID of the speaker to update
            name: New name (optional)
            face_encoding: New face encoding (optional)
            voice_profile: New voice profile (optional)
            notes: New notes (optional)
            
        Returns:
            bool: True if successful
        """
        try:
            self.ensure_connection()
            cursor = self.conn.cursor()
            
            # Build update query dynamically based on provided parameters
            update_items = []
            params = []
            
            if name is not None:
                update_items.append("name = ?")
                params.append(name)
            
            if face_encoding is not None:
                update_items.append("face_encoding = ?")
                params.append(face_encoding)
                
            if voice_profile is not None:
                update_items.append("voice_profile = ?")
                params.append(voice_profile)
                
            if notes is not None:
                update_items.append("notes = ?")
                params.append(notes)
            
            # Add updated_at timestamp
            update_items.append("updated_at = CURRENT_TIMESTAMP")
            
            # If no updates, return True
            if not update_items:
                return True
                
            # Add speaker_id to params
            params.append(speaker_id)
            
            # Execute update
            cursor.execute(
                f"""
                UPDATE speakers
                SET {', '.join(update_items)}
                WHERE id = ?
                """,
                params
            )
            
            self.conn.commit()
            rows_affected = cursor.rowcount
            
            if rows_affected > 0:
                logger.info(f"Updated speaker with ID {speaker_id}")
                return True
            else:
                logger.warning(f"No speaker found with ID {speaker_id}")
                return False
                
        except sqlite3.Error as e:
            logger.error(f"Error updating speaker: {e}")
            return False
    
    def delete_speaker(self, speaker_id):
        """
        Delete a speaker from the database
        
        Args:
            speaker_id: ID of the speaker to delete
            
        Returns:
            bool: True if successful
        """
        try:
            self.ensure_connection()
            cursor = self.conn.cursor()
            
            cursor.execute("DELETE FROM speakers WHERE id = ?", (speaker_id,))
            self.conn.commit()
            
            rows_affected = cursor.rowcount
            
            if rows_affected > 0:
                logger.info(f"Deleted speaker with ID {speaker_id}")
                return True
            else:
                logger.warning(f"No speaker found with ID {speaker_id}")
                return False
                
        except sqlite3.Error as e:
            logger.error(f"Error deleting speaker: {e}")
            return False
    
    def start_conversation(self, title=None, location=None, context=None, metadata=None):
        """
        Start a new conversation in the database
        
        Args:
            title: Conversation title
            location: Conversation location
            context: Additional context
            metadata: JSON-serializable metadata dict
            
        Returns:
            int: ID of the new conversation
        """
        try:
            self.ensure_connection()
            cursor = self.conn.cursor()
            
            metadata_str = json.dumps(metadata) if metadata else None
            
            cursor.execute(
                """
                INSERT INTO conversations (start_time, title, location, context, metadata)
                VALUES (CURRENT_TIMESTAMP, ?, ?, ?, ?)
                """,
                (title, location, context, metadata_str)
            )
            
            self.conn.commit()
            conversation_id = cursor.lastrowid
            logger.info(f"Started new conversation with ID {conversation_id}")
            return conversation_id
            
        except sqlite3.Error as e:
            logger.error(f"Error starting conversation: {e}")
            return None
    
    def end_conversation(self, conversation_id):
        """
        End a conversation by setting its end time
        
        Args:
            conversation_id: ID of the conversation to end
            
        Returns:
            bool: True if successful
        """
        try:
            self.ensure_connection()
            cursor = self.conn.cursor()
            
            cursor.execute(
                """
                UPDATE conversations
                SET end_time = CURRENT_TIMESTAMP
                WHERE id = ? AND end_time IS NULL
                """,
                (conversation_id,)
            )
            
            self.conn.commit()
            rows_affected = cursor.rowcount
            
            if rows_affected > 0:
                logger.info(f"Ended conversation with ID {conversation_id}")
                return True
            else:
                logger.warning(f"No active conversation found with ID {conversation_id}")
                return False
                
        except sqlite3.Error as e:
            logger.error(f"Error ending conversation: {e}")
            return False
    
    def add_utterance(self, conversation_id, speaker_id=None, speaker_name=None, 
                     text=None, start_time=None, end_time=None, confidence=None, 
                     audio_path=None):
        """
        Add a new utterance to a conversation
        
        Args:
            conversation_id: ID of the conversation
            speaker_id: ID of the speaker (optional)
            speaker_name: Name of the speaker (optional)
            text: Transcribed text of the utterance
            start_time: Start time of the utterance (default: current time)
            end_time: End time of the utterance (default: current time)
            confidence: Confidence score of the transcription (0-1)
            audio_path: Path to the audio file (optional)
            
        Returns:
            int: ID of the added utterance
        """
        try:
            self.ensure_connection()
            cursor = self.conn.cursor()
            
            # Get current time for timestamps if not provided
            now = datetime.now()
            if start_time is None:
                start_time = now
            if end_time is None:
                end_time = now
                
            # If only speaker name is provided, try to get speaker ID
            if speaker_id is None and speaker_name is not None:
                speaker = self.get_speaker(name=speaker_name)
                if speaker:
                    speaker_id = speaker["id"]
            
            cursor.execute(
                """
                INSERT INTO utterances (
                    conversation_id, speaker_id, speaker_name, 
                    text, start_time, end_time, 
                    confidence, audio_path
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id, speaker_id, speaker_name,
                    text, start_time, end_time,
                    confidence, audio_path
                )
            )
            
            self.conn.commit()
            utterance_id = cursor.lastrowid
            logger.debug(f"Added utterance with ID {utterance_id} to conversation {conversation_id}")
            return utterance_id
            
        except sqlite3.Error as e:
            logger.error(f"Error adding utterance: {e}")
            return None
    
    def get_conversation(self, conversation_id):
        """
        Get a conversation by ID, including all utterances
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            dict: Conversation data with utterances
        """
        try:
            self.ensure_connection()
            cursor = self.conn.cursor()
            
            # Get conversation data
            cursor.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
            conversation = cursor.fetchone()
            
            if not conversation:
                logger.warning(f"Conversation {conversation_id} not found")
                return None
                
            # Convert to dictionary
            conversation_dict = dict(conversation)
            
            # Parse metadata JSON if present
            if conversation_dict.get("metadata"):
                try:
                    conversation_dict["metadata"] = json.loads(conversation_dict["metadata"])
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in metadata for conversation {conversation_id}")
            
            # Get utterances
            cursor.execute(
                """
                SELECT * FROM utterances
                WHERE conversation_id = ?
                ORDER BY start_time
                """,
                (conversation_id,)
            )
            
            utterances = [dict(row) for row in cursor.fetchall()]
            conversation_dict["utterances"] = utterances
            
            return conversation_dict
            
        except sqlite3.Error as e:
            logger.error(f"Error getting conversation: {e}")
            return None
    
    def get_recent_conversations(self, limit=10):
        """
        Get recent conversations
        
        Args:
            limit: Maximum number of conversations to return
            
        Returns:
            list: List of recent conversations without utterances
        """
        try:
            self.ensure_connection()
            cursor = self.conn.cursor()
            
            cursor.execute(
                """
                SELECT * FROM conversations
                ORDER BY start_time DESC
                LIMIT ?
                """,
                (limit,)
            )
            
            conversations = []
            for row in cursor.fetchall():
                conv = dict(row)
                
                # Parse metadata JSON if present
                if conv.get("metadata"):
                    try:
                        conv["metadata"] = json.loads(conv["metadata"])
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in metadata for conversation {conv['id']}")
                
                # Count utterances
                cursor.execute(
                    "SELECT COUNT(*) FROM utterances WHERE conversation_id = ?",
                    (conv["id"],)
                )
                conv["utterance_count"] = cursor.fetchone()[0]
                
                conversations.append(conv)
            
            return conversations
            
        except sqlite3.Error as e:
            logger.error(f"Error getting recent conversations: {e}")
            return []
    
    def export_conversation(self, conversation_id, format="json"):
        """
        Export a conversation in the specified format
        
        Args:
            conversation_id: ID of the conversation
            format: Export format (json, text)
            
        Returns:
            str or dict: Exported conversation
        """
        conversation = self.get_conversation(conversation_id)
        
        if not conversation:
            return None
            
        if format == "json":
            return conversation
        elif format == "text":
            # Format as text
            lines = [
                f"Conversation: {conversation.get('title', 'Untitled')}",
                f"Date: {conversation.get('start_time', '')} to {conversation.get('end_time', 'ongoing')}",
                f"Location: {conversation.get('location', 'Unknown')}",
                f"Context: {conversation.get('context', '')}",
                ""
            ]
            
            for utterance in conversation.get("utterances", []):
                speaker = utterance.get("speaker_name", "Unknown")
                timestamp = utterance.get("start_time", "")
                text = utterance.get("text", "")
                lines.append(f"[{timestamp}] {speaker}: {text}")
                
            return "\n".join(lines)
        else:
            logger.error(f"Unsupported export format: {format}")
            return None
    
    def record_metric(self, metric_name, metric_value, context=None):
        """
        Record a system metric
        
        Args:
            metric_name: Name of the metric
            metric_value: Value of the metric
            context: Additional context information
            
        Returns:
            bool: True if successful
        """
        try:
            self.ensure_connection()
            cursor = self.conn.cursor()
            
            cursor.execute(
                """
                INSERT INTO metrics (metric_name, metric_value, context)
                VALUES (?, ?, ?)
                """,
                (metric_name, metric_value, context)
            )
            
            self.conn.commit()
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Error recording metric: {e}")
            return False
    
    def get_metrics(self, metric_name=None, start_time=None, end_time=None, limit=100):
        """
        Get system metrics
        
        Args:
            metric_name: Filter by metric name (optional)
            start_time: Start time filter (optional)
            end_time: End time filter (optional)
            limit: Maximum number of records to return
            
        Returns:
            list: List of metric records
        """
        try:
            self.ensure_connection()
            cursor = self.conn.cursor()
            
            query = "SELECT * FROM metrics"
            conditions = []
            params = []
            
            if metric_name:
                conditions.append("metric_name = ?")
                params.append(metric_name)
                
            if start_time:
                conditions.append("timestamp >= ?")
                params.append(start_time)
                
            if end_time:
                conditions.append("timestamp <= ?")
                params.append(end_time)
                
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
                
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
            
        except sqlite3.Error as e:
            logger.error(f"Error getting metrics: {e}")
            return []
    
    def vacuum_database(self):
        """
        Run VACUUM on the database to optimize storage
        
        Returns:
            bool: True if successful
        """
        try:
            self.ensure_connection()
            self.conn.execute("VACUUM")
            logger.info("Database vacuum completed")
            return True
        except sqlite3.Error as e:
            logger.error(f"Error vacuuming database: {e}")
            return False
            
    def __enter__(self):
        """Support for context manager"""
        self.ensure_connection()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support for context manager"""
        self.close()