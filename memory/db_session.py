import sqlite3
import os

DB_PATH = "robotics_brain.db"

def init_db():
    """Initializes the SQLite tables for storing object coordinates and task execution history."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Table for physical objects detected in coordinates space
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS workspace_objects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        item_name TEXT UNIQUE NOT NULL,
        shape TEXT NOT NULL,
        color_hex TEXT NOT NULL,
        loc_x REAL NOT NULL,
        loc_y REAL NOT NULL,
        loc_z REAL NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # 2. Table for task planner execution records (historic runs logs)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS task_history (
        task_id TEXT PRIMARY KEY,
        raw_command TEXT NOT NULL,
        interpreted_goal TEXT NOT NULL,
        target_object TEXT NOT NULL,
        destination TEXT NOT NULL,
        execution_status TEXT CHECK(execution_status IN ('PENDING', 'ACTIVE', 'SUCCESS', 'FAILED')),
        execution_time_sec REAL,
        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 3. Table for robot trajectory logging (motor angle targets state tracker)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS trajectory_telemetry (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT NOT NULL,
        sequence_step INTEGER NOT NULL,
        joint_1_deg REAL NOT NULL,
        joint_2_deg REAL NOT NULL,
        joint_3_deg REAL NOT NULL,
        vacuum_state INTEGER CHECK(vacuum_state IN (0, 1)),
        FOREIGN KEY(task_id) REFERENCES task_history(task_id)
    );
    """)
    
    # Seed default workspace coordinates list so the model starts with populated targets
    default_objects = [
        ("red cube", "cube", "#ef4444", 80.0, 150.0, 10.0),
        ("blue box", "box", "#3b82f6", -100.0, 180.0, 20.0),
        ("green sphere", "sphere", "#22c55e", 120.0, 120.0, 10.0),
        ("yellow container", "box", "#eab308", -60.0, 140.0, 20.0),
        ("orange pyramid", "pyramid", "#f97316", 40.0, 190.0, 15.0)
    ]
    
    for name, shape, color, x, y, z in default_objects:
        cursor.execute("""
        INSERT INTO workspace_objects (item_name, shape, color_hex, loc_x, loc_y, loc_z)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(item_name) DO UPDATE SET loc_x=excluded.loc_x, loc_y=excluded.loc_y, loc_z=excluded.loc_z;
        """, (name, shape, color, x, y, z))
        
    conn.commit()
    conn.close()

def get_db_connection():
    """Generates an active connection object to the database."""
    return sqlite3.connect(DB_PATH)
