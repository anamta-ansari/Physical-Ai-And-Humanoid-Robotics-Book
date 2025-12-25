import psycopg2
from psycopg2.extras import RealDictCursor
from config.settings import settings
from typing import Optional, List
import json
from datetime import datetime
from models.chat import ChatSession


class HistoryService:
    """Service for managing chat session history using Neon Postgres."""
    
    def __init__(self):
        if settings.NEON_DB_URL:
            try:
                # Parse the database URL to extract connection parameters
                import urllib.parse
                url = urllib.parse.urlparse(settings.NEON_DB_URL)

                self.connection = psycopg2.connect(
                    host=url.hostname,
                    port=url.port,
                    database=url.path[1:],  # Remove leading slash
                    user=url.username,
                    password=url.password
                )
                self.create_session_table()
                self.initialized = True
                print("History service initialized successfully")
            except Exception as e:
                print(f"Error connecting to Neon Postgres: {str(e)}")
                print("Warning: Session history will not be persisted until database connection is established.")
                self.connection = None
                self.initialized = False
        else:
            self.connection = None
            self.initialized = False
            print("Warning: Neon Postgres URL not provided. Session history will not be persisted.")
    
    def create_session_table(self):
        """Create the sessions table if it doesn't exist."""
        if not self.connection:
            return
            
        with self.connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id VARCHAR(255) PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_interaction TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    active BOOLEAN DEFAULT TRUE,
                    metadata JSONB
                )
            """)
            self.connection.commit()
    
    def create_session(self, session_id: str, metadata: Optional[dict] = None) -> ChatSession:
        """Create a new chat session."""
        if not self.connection:
            # Return an in-memory session if no database connection
            return ChatSession(session_id=session_id)
            
        with self.connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO sessions (session_id, metadata)
                VALUES (%s, %s)
                ON CONFLICT (session_id) DO UPDATE
                SET active = TRUE, last_interaction = CURRENT_TIMESTAMP
            """, (session_id, json.dumps(metadata or {})))
            self.connection.commit()
            
            return self.get_session(session_id)
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a chat session by ID."""
        if not self.connection:
            # Return a basic session if no database connection
            return ChatSession(session_id=session_id)
            
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT session_id, created_at, last_interaction, active
                FROM sessions
                WHERE session_id = %s
            """, (session_id,))
            row = cursor.fetchone()
            
            if row:
                return ChatSession(
                    session_id=row['session_id'],
                    created_at=row['created_at'],
                    last_interaction=row['last_interaction'],
                    active=row['active']
                )
            return None
    
    def update_session(self, session_id: str, active: Optional[bool] = None):
        """Update a chat session."""
        if not self.connection:
            return
            
        with self.connection.cursor() as cursor:
            if active is not None:
                cursor.execute("""
                    UPDATE sessions
                    SET active = %s, last_interaction = CURRENT_TIMESTAMP
                    WHERE session_id = %s
                """, (active, session_id))
            else:
                cursor.execute("""
                    UPDATE sessions
                    SET last_interaction = CURRENT_TIMESTAMP
                    WHERE session_id = %s
                """, (session_id,))
            self.connection.commit()
    
    def delete_session(self, session_id: str):
        """Delete a chat session."""
        if not self.connection:
            return
            
        with self.connection.cursor() as cursor:
            cursor.execute("DELETE FROM sessions WHERE session_id = %s", (session_id,))
            self.connection.commit()