# src/modules/database/database_manager.py

import asyncpg        # asyncpg lets us talk to PostgreSQL asynchronously
import uuid           # for generating unique IDs
import json           # for storing lists/dicts as JSON in the DB
import os
from typing import List, Dict, Any, Optional

# Why asyncpg instead of regular psycopg2?
# FastAPI is async (uses "await"). If we use a blocking DB driver,
# the whole server freezes while waiting for the DB.
# asyncpg lets the server handle other requests while waiting for DB.


class DatabaseManager:
    def __init__(self):
        # pool = a group of ready-made DB connections.
        # Instead of connecting/disconnecting for every request (slow),
        # we keep a pool of open connections and reuse them.
        self.pool = None

    async def initialize(self):
        """Call this once when the app starts up."""

        # Build the connection string from environment variables.
        # We read from env so passwords aren't hardcoded in source code.
        database_url = (
            f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
            f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT', '5432')}"
            f"/{os.getenv('DB_NAME')}"
        )

        # Create the connection pool.
        # min_size=2 means always keep 2 connections warm.
        # max_size=10 means never open more than 10 at once.
        self.pool = await asyncpg.create_pool(
            database_url,
            min_size=2,
            max_size=10
        )

        # Create tables if they don't exist yet.
        await self._create_tables()
        print("✅ Database connected and tables ready")

    async def _create_tables(self):
        """Create all tables. IF NOT EXISTS means it's safe to run repeatedly."""

        # We use "async with self.pool.acquire() as conn" to borrow
        # a connection from the pool, use it, then return it automatically.
        async with self.pool.acquire() as conn:
            await conn.execute('''

                -- SESSIONS TABLE
                -- One session = one "job" the user is doing
                -- e.g. "Generate test cases for my login feature"
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id  VARCHAR(255) PRIMARY KEY,
                    user_id     VARCHAR(255) NOT NULL,
                    project_name VARCHAR(255),
                    user_prompt  TEXT,
                    status       VARCHAR(50) DEFAULT 'created',
                    created_at   TIMESTAMP DEFAULT NOW(),
                    updated_at   TIMESTAMP DEFAULT NOW()
                );

                -- REQUIREMENTS TABLE
                -- Stores the requirements the AI analyzed.
                -- "session_id REFERENCES sessions" = Foreign Key.
                -- ON DELETE CASCADE = if the session is deleted,
                --   all its requirements are automatically deleted too.
                CREATE TABLE IF NOT EXISTS requirements (
                    id               VARCHAR(255) PRIMARY KEY,
                    session_id       VARCHAR(255) REFERENCES sessions(session_id)
                                     ON DELETE CASCADE,
                    original_content TEXT NOT NULL,
                    edited_content   TEXT,
                    requirement_type VARCHAR(50) DEFAULT 'functional',
                    status           VARCHAR(20) DEFAULT 'active',
                    created_at       TIMESTAMP DEFAULT NOW()
                );

                -- TEST CASES TABLE
                -- Stores AI-generated test cases.
                -- test_steps is JSONB = stores a JSON list of steps.
                -- JSONB is PostgreSQL's way of storing structured data
                -- inside a column (like a list inside a cell).
                CREATE TABLE IF NOT EXISTS test_cases (
                    id               VARCHAR(255) PRIMARY KEY,
                    session_id       VARCHAR(255) REFERENCES sessions(session_id)
                                     ON DELETE CASCADE,
                    test_name        VARCHAR(255) NOT NULL,
                    test_description TEXT,
                    test_steps       JSONB,
                    expected_results TEXT,
                    test_type        VARCHAR(50) DEFAULT 'functional',
                    priority         VARCHAR(10) DEFAULT 'medium',
                    status           VARCHAR(20) DEFAULT 'active',
                    created_at       TIMESTAMP DEFAULT NOW()
                );

                -- INDEXES for speed.
                -- Without an index, "give me all requirements for session X"
                -- scans EVERY row. With an index, it jumps straight to them.
                CREATE INDEX IF NOT EXISTS idx_req_session
                    ON requirements(session_id);
                CREATE INDEX IF NOT EXISTS idx_tc_session
                    ON test_cases(session_id);

            ''')

    # ─────────────────────────────────────────────
    # SESSION METHODS
    # ─────────────────────────────────────────────

    async def create_session(
        self,
        session_id: str,
        user_id: str,
        project_name: str,
        user_prompt: str
    ):
        """Insert a new session row."""
        async with self.pool.acquire() as conn:
            # $1, $2, $3, $4 are placeholders — asyncpg fills them in.
            # This prevents SQL injection attacks.
            # Never use f-strings to build SQL queries with user input!
            await conn.execute('''
                INSERT INTO sessions
                    (session_id, user_id, project_name, user_prompt, status)
                VALUES
                    ($1, $2, $3, $4, 'created')
            ''', session_id, user_id, project_name, user_prompt)

    async def update_session_status(self, session_id: str, status: str):
        """Update what stage the session is at."""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                UPDATE sessions
                SET status = $1, updated_at = NOW()
                WHERE session_id = $2
            ''', status, session_id)

    async def get_session(self, session_id: str) -> Optional[Dict]:
        """Fetch a single session by ID. Returns None if not found."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow('''
                SELECT * FROM sessions WHERE session_id = $1
            ''', session_id)

            # asyncpg returns a Record object, not a dict.
            # dict(row) converts it so we can return/serialize it.
            return dict(row) if row else None

    # ─────────────────────────────────────────────
    # REQUIREMENTS METHODS
    # ─────────────────────────────────────────────

    async def save_requirements(
        self,
        session_id: str,
        requirements: List[str]
    ):
        """Save a list of requirement strings for a session."""
        async with self.pool.acquire() as conn:
            for req_text in requirements:
                # Generate a unique ID for each requirement.
                # uuid4() = random unique string like "3f2e1a..."
                req_id = f"{session_id}_req_{uuid.uuid4().hex[:8]}"

                await conn.execute('''
                    INSERT INTO requirements
                        (id, session_id, original_content, requirement_type)
                    VALUES
                        ($1, $2, $3, 'functional')
                    ON CONFLICT (id) DO NOTHING
                ''', req_id, session_id, req_text)
                # ON CONFLICT DO NOTHING = if this ID already exists,
                # skip silently instead of crashing.

    async def get_requirements(self, session_id: str) -> List[Dict]:
        """Get all active requirements for a session."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT * FROM requirements
                WHERE session_id = $1
                  AND status != 'deleted'
                ORDER BY created_at ASC
            ''', session_id)

            return [dict(row) for row in rows]

    # ─────────────────────────────────────────────
    # TEST CASES METHODS
    # ─────────────────────────────────────────────

    async def save_test_cases(
        self,
        session_id: str,
        test_cases: List[Dict]
    ):
        """Save a list of test case dicts for a session."""
        async with self.pool.acquire() as conn:
            for i, tc in enumerate(test_cases):
                tc_id = f"{session_id}_tc_{uuid.uuid4().hex[:8]}"

                await conn.execute('''
                    INSERT INTO test_cases (
                        id, session_id, test_name, test_description,
                        test_steps, expected_results, test_type, priority
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (id) DO NOTHING
                ''',
                    tc_id,
                    session_id,
                    tc.get('test_name', f'Test Case {i+1}'),
                    tc.get('test_description', ''),
                    # json.dumps converts a Python list → JSON string
                    # so PostgreSQL can store it in the JSONB column
                    json.dumps(tc.get('test_steps', [])),
                    tc.get('expected_results', ''),
                    tc.get('test_type', 'functional'),
                    tc.get('priority', 'medium')
                )

    async def get_test_cases(self, session_id: str) -> List[Dict]:
        """Get all active test cases for a session."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT * FROM test_cases
                WHERE session_id = $1
                  AND status = 'active'
                ORDER BY created_at ASC
            ''', session_id)

            result = []
            for row in rows:
                row_dict = dict(row)
                # test_steps was stored as a JSON string.
                # Parse it back to a Python list when we read it.
                if isinstance(row_dict.get('test_steps'), str):
                    try:
                        row_dict['test_steps'] = json.loads(row_dict['test_steps'])
                    except Exception:
                        row_dict['test_steps'] = []
                result.append(row_dict)

            return result

    async def close(self):
        """Clean up the connection pool when the app shuts down."""
        if self.pool:
            await self.pool.close()


# A single shared instance — imported by the rest of the app.
# This way there's only ONE pool, not one per request.
db_manager = DatabaseManager()