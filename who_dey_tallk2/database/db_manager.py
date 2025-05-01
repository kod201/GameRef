#!/usr/bin/env python3
"""
Database Manager for Who Dey Tallk 2

Handles database operations for storing speakers, conversations, and statistics
"""
import os
import sqlite3
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database manager for Who Dey Tallk system"""
    
    def __init__(self, db_path, create_if_missing=True):
        """
        Initialize the database manager
        
        Args:
            db_path: Path to SQLite database file
            create_if_missing: Whether to create the database if it doesn't exist
        """
        self.db_path = Path(db_path)
        self.connection = None
        self.lock = threading.RLock()
        
        # Create database if requested
        if create_if_missing:
            self._ensure_db_exists()
    
    def _ensure_db_exists(self):
        """Ensure database exists and has required tables"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Connect to database (creates it if it doesn't exist)
            conn = self._get_connection()
            
            # Create tables if they don't exist
            with conn:
                cursor = conn.cursor()
                
                # Create speakers table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS speakers (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        first_seen_timestamp TEXT NOT NULL,
                        last_seen_timestamp TEXT NOT NULL
                    )
                ''')
                
                # Create conversations table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        speaker_id TEXT NOT NULL,
                        text TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        timestamp TEXT NOT NULL
                    )
                ''')
                
                # Create statistics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS statistics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        video_fps REAL,
                        audio_level REAL,
                        speakers_detected INTEGER,
                        faces_detected INTEGER,
                        system_load REAL,
                        recognition_rate REAL
                    )
                ''')
                
                # Add indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_conversations_speaker ON conversations (speaker_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations (timestamp)')
                
                logger.info("Database tables created/verified successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def _get_connection(self):
        """
        Get database connection (creating it if necessary)
        
        Returns:
            sqlite3.Connection: Database connection
        """
        if self.connection is None:
            try:
                self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
                self.connection.row_factory = sqlite3.Row  # Access rows by column name
            except Exception as e:
                logger.error(f"Error connecting to database: {e}")
                raise
        
        return self.connection
    
    def close(self):
        """Close the database connection"""
        with self.lock:
            if self.connection:
                try:
                    self.connection.close()
                    self.connection = None
                    logger.info("Database connection closed")
                except Exception as e:
                    logger.error(f"Error closing database connection: {e}")
    
    def add_speaker(self, speaker_id, name):
        """
        Add a new speaker or update an existing one
        
        Args:
            speaker_id: Unique identifier for the speaker
            name: Name of the speaker
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.lock:
                conn = self._get_connection()
                now = datetime.now().isoformat()
                
                with conn:
                    cursor = conn.cursor()
                    
                    # Check if speaker already exists
                    cursor.execute("SELECT id FROM speakers WHERE id = ?", (speaker_id,))
                    result = cursor.fetchone()
                    
                    if result:
                        # Update existing speaker
                        cursor.execute(
                            "UPDATE speakers SET name = ?, last_seen_timestamp = ? WHERE id = ?",
                            (name, now, speaker_id)
                        )
                        logger.debug(f"Updated speaker: {speaker_id} ({name})")
                    else:
                        # Add new speaker
                        cursor.execute(
                            "INSERT INTO speakers (id, name, first_seen_timestamp, last_seen_timestamp) VALUES (?, ?, ?, ?)",
                            (speaker_id, name, now, now)
                        )
                        logger.info(f"Added new speaker: {speaker_id} ({name})")
                
                return True
                
        except Exception as e:
            logger.error(f"Error adding/updating speaker: {e}")
            return False
    
    def get_speaker(self, speaker_id):
        """
        Get speaker information
        
        Args:
            speaker_id: Unique identifier for the speaker
            
        Returns:
            dict: Speaker information or None if not found
        """
        try:
            with self.lock:
                conn = self._get_connection()
                
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM speakers WHERE id = ?", (speaker_id,))
                result = cursor.fetchone()
                
                if result:
                    return dict(result)
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting speaker: {e}")
            return None
    
    def list_speakers(self):
        """
        List all speakers
        
        Returns:
            list: List of speaker dictionaries
        """
        try:
            with self.lock:
                conn = self._get_connection()
                
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM speakers ORDER BY last_seen_timestamp DESC")
                results = cursor.fetchall()
                
                return [dict(row) for row in results]
                
        except Exception as e:
            logger.error(f"Error listing speakers: {e}")
            return []
    
    def add_conversation(self, speaker_id, text, confidence, timestamp=None):
        """
        Add a conversation entry
        
        Args:
            speaker_id: ID of the speaker
            text: Transcribed text
            confidence: Confidence score (0.0-1.0)
            timestamp: Timestamp (ISO format) or None for current time
            
        Returns:
            int: ID of the new conversation entry or None if failed
        """
        try:
            with self.lock:
                conn = self._get_connection()
                
                if timestamp is None:
                    timestamp = datetime.now().isoformat()
                
                # Ensure speaker exists (for unknown speakers, we still want an entry)
                if speaker_id != "unknown":
                    self.add_speaker(speaker_id, speaker_id)
                
                with conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO conversations (speaker_id, text, confidence, timestamp) VALUES (?, ?, ?, ?)",
                        (speaker_id, text, confidence, timestamp)
                    )
                    
                    # Get the ID of the new row
                    conversation_id = cursor.lastrowid
                    
                    logger.debug(f"Added conversation entry: {speaker_id} said \"{text}\" (ID: {conversation_id})")
                    
                    return conversation_id
                
        except Exception as e:
            logger.error(f"Error adding conversation: {e}")
            return None
    
    def get_conversation(self, conversation_id):
        """
        Get a conversation entry by ID
        
        Args:
            conversation_id: ID of the conversation entry
            
        Returns:
            dict: Conversation entry or None if not found
        """
        try:
            with self.lock:
                conn = self._get_connection()
                
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
                result = cursor.fetchone()
                
                if result:
                    return dict(result)
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting conversation: {e}")
            return None
    
    def get_speaker_conversations(self, speaker_id, limit=50, offset=0):
        """
        Get conversations for a specific speaker
        
        Args:
            speaker_id: ID of the speaker
            limit: Maximum number of results to return
            offset: Offset for pagination
            
        Returns:
            list: List of conversation dictionaries
        """
        try:
            with self.lock:
                conn = self._get_connection()
                
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM conversations WHERE speaker_id = ? ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                    (speaker_id, limit, offset)
                )
                results = cursor.fetchall()
                
                return [dict(row) for row in results]
                
        except Exception as e:
            logger.error(f"Error getting speaker conversations: {e}")
            return []
    
    def search_conversations(self, query, limit=50, offset=0):
        """
        Search conversations by text
        
        Args:
            query: Text to search for
            limit: Maximum number of results to return
            offset: Offset for pagination
            
        Returns:
            list: List of conversation dictionaries
        """
        try:
            with self.lock:
                conn = self._get_connection()
                
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM conversations WHERE text LIKE ? ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                    (f"%{query}%", limit, offset)
                )
                results = cursor.fetchall()
                
                return [dict(row) for row in results]
                
        except Exception as e:
            logger.error(f"Error searching conversations: {e}")
            return []
    
    def get_recent_conversations(self, hours=24, limit=50):
        """
        Get recent conversations
        
        Args:
            hours: Number of hours to look back
            limit: Maximum number of results to return
            
        Returns:
            list: List of conversation dictionaries
        """
        try:
            with self.lock:
                conn = self._get_connection()
                
                # Calculate timestamp for filtering
                start_time = (datetime.now() - timedelta(hours=hours)).isoformat()
                
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM conversations WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT ?",
                    (start_time, limit)
                )
                results = cursor.fetchall()
                
                return [dict(row) for row in results]
                
        except Exception as e:
            logger.error(f"Error getting recent conversations: {e}")
            return []
    
    def add_system_statistics(self, video_fps=0.0, audio_level=0.0, speakers_detected=0,
                             faces_detected=0, system_load=0.0, recognition_rate=0.0):
        """
        Add system statistics entry
        
        Args:
            video_fps: Video frames per second
            audio_level: Audio level (0.0-1.0)
            speakers_detected: Number of speakers detected
            faces_detected: Number of faces detected
            system_load: System load (CPU usage %)
            recognition_rate: Recognition success rate (0.0-1.0)
            
        Returns:
            int: ID of the new statistics entry or None if failed
        """
        try:
            with self.lock:
                conn = self._get_connection()
                timestamp = datetime.now().isoformat()
                
                with conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        '''
                        INSERT INTO statistics 
                        (timestamp, video_fps, audio_level, speakers_detected, faces_detected, system_load, recognition_rate)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''',
                        (timestamp, video_fps, audio_level, speakers_detected, faces_detected, system_load, recognition_rate)
                    )
                    
                    # Get the ID of the new row
                    stats_id = cursor.lastrowid
                    
                    logger.debug(f"Added statistics entry (ID: {stats_id})")
                    
                    return stats_id
                
        except Exception as e:
            logger.error(f"Error adding statistics: {e}")
            return None
    
    def clean_old_data(self, days=30):
        """
        Clean up old data
        
        Args:
            days: Remove data older than this many days
            
        Returns:
            tuple: (conversations_deleted, statistics_deleted)
        """
        try:
            with self.lock:
                conn = self._get_connection()
                
                # Calculate cutoff timestamp
                cutoff_time = (datetime.now() - timedelta(days=days)).isoformat()
                
                with conn:
                    cursor = conn.cursor()
                    
                    # Delete old conversations
                    cursor.execute("DELETE FROM conversations WHERE timestamp < ?", (cutoff_time,))
                    conversations_deleted = cursor.rowcount
                    
                    # Delete old statistics
                    cursor.execute("DELETE FROM statistics WHERE timestamp < ?", (cutoff_time,))
                    statistics_deleted = cursor.rowcount
                    
                    logger.info(f"Cleaned up old data: {conversations_deleted} conversations, {statistics_deleted} statistics entries")
                    
                    return (conversations_deleted, statistics_deleted)
                
        except Exception as e:
            logger.error(f"Error cleaning old data: {e}")
            return (0, 0)
    
    def get_statistics_summary(self, hours=24):
        """
        Get system statistics summary
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            dict: Summary statistics
        """
        try:
            with self.lock:
                conn = self._get_connection()
                
                # Calculate timestamp for filtering
                start_time = (datetime.now() - timedelta(hours=hours)).isoformat()
                
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT 
                        AVG(video_fps) as avg_fps,
                        AVG(audio_level) as avg_audio_level,
                        AVG(system_load) as avg_system_load,
                        MAX(speakers_detected) as max_speakers,
                        MAX(faces_detected) as max_faces,
                        AVG(recognition_rate) as avg_recognition_rate,
                        COUNT(*) as total_entries
                    FROM statistics
                    WHERE timestamp >= ?
                    ''',
                    (start_time,)
                )
                result = cursor.fetchone()
                
                if result:
                    return dict(result)
                
                return {
                    "avg_fps": 0.0,
                    "avg_audio_level": 0.0,
                    "avg_system_load": 0.0,
                    "max_speakers": 0,
                    "max_faces": 0,
                    "avg_recognition_rate": 0.0,
                    "total_entries": 0
                }
                
        except Exception as e:
            logger.error(f"Error getting statistics summary: {e}")
            return {}