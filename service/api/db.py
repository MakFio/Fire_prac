import os
import json
import aiosqlite
from datetime import datetime

DB_PATH = os.getenv("DB_PATH", "/app/fire.db")

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS requests (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                filename   TEXT,
                model_name TEXT,
                detections TEXT,
                latency_ms REAL,
                alarm      INTEGER,
                timestamp  TEXT
            )
        """)
        await db.commit()

async def log_request(filename, model_name, detections, latency_ms, alarm):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO requests (filename,model_name,detections,latency_ms,alarm,timestamp) "
            "VALUES (?,?,?,?,?,?)",
            (filename, model_name, json.dumps(detections, ensure_ascii=False),
             latency_ms, int(alarm), datetime.now().isoformat()),
        )
        await db.commit()

async def get_stats():
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT COUNT(*), AVG(latency_ms), SUM(alarm) FROM requests") as cur:
            row = await cur.fetchone()
    return {
        "total_requests": row[0] or 0,
        "avg_latency_ms": round(row[1] or 0.0, 2),
        "total_alarms":   row[2] or 0,
    }

async def get_history(limit: int = 50):
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT filename,model_name,detections,latency_ms,alarm,timestamp "
            "FROM requests ORDER BY id DESC LIMIT ?", (limit,)
        ) as cur:
            rows = await cur.fetchall()
    return [
        {"filename": r[0], "model": r[1], "detections": json.loads(r[2]),
         "latency_ms": r[3], "alarm": bool(r[4]), "timestamp": r[5]}
        for r in rows
    ]