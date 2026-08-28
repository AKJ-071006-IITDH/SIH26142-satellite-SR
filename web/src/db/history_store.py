import sqlite3
import json
import uuid
import time
from pathlib import Path
from typing import List, Dict, Optional


class HistoryStore:
    def __init__(self, db_path="data/history.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True, parents=True)
        self._init_db()
        self._migrate_schema()   # NEW -- adds new columns to an existing
                                    # DB file safely, instead of requiring
                                    # you to delete data/history.db and
                                    # lose all your existing demo runs

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def close(self):
        import gc
        gc.collect()

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id TEXT PRIMARY KEY,
                    timestamp REAL,
                    date_str TEXT,
                    name TEXT,
                    lr_b64 TEXT,
                    sr_b64 TEXT,
                    uncertainty_b64 TEXT,
                    ndvi_b64 TEXT,
                    metrics_json TEXT,
                    params_json TEXT
                )
            """)
            conn.commit()

    def _migrate_schema(self):
        """
        Adds model_id and gt_b64 columns if they don't already exist.
        SQLite has no "ADD COLUMN IF NOT EXISTS", so we check the
        existing columns first and only add what's missing -- safe to
        run every time the app starts, on a fresh DB or an existing one.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(history)")
            existing_columns = {row[1] for row in cursor.fetchall()}

            if "model_id" not in existing_columns:
                cursor.execute("ALTER TABLE history ADD COLUMN model_id TEXT DEFAULT 'phase2'")
                print("Migrated history.db: added model_id column")
            if "gt_b64" not in existing_columns:
                cursor.execute("ALTER TABLE history ADD COLUMN gt_b64 TEXT")
                print("Migrated history.db: added gt_b64 column")
            conn.commit()

    def add_entry(self, name: str, lr_b64: str, sr_b64: str, uncertainty_b64: str,
                  ndvi_b64: str, metrics: dict, params: dict,
                  model_id: str = "phase2", gt_b64: str = None) -> str:
        entry_id = str(uuid.uuid4())
        now = time.time()
        date_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
        metrics_json = json.dumps(metrics)
        params_json = json.dumps(params)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO history
                (id, timestamp, date_str, name, model_id, lr_b64, sr_b64, gt_b64,
                 uncertainty_b64, ndvi_b64, metrics_json, params_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (entry_id, now, date_str, name, model_id, lr_b64, sr_b64, gt_b64,
                  uncertainty_b64, ndvi_b64, metrics_json, params_json))
            conn.commit()

        print(f"Saved run '{name}' (model: {model_id}) to history DB (ID: {entry_id[:8]})")
        return entry_id

    def _row_to_dict(self, r, columns) -> Dict:
        d = dict(zip(columns, r))
        d["metrics"] = json.loads(d.pop("metrics_json")) if d.get("metrics_json") else {}
        d["params"] = json.loads(d.pop("params_json")) if d.get("params_json") else {}
        return d

    def list_entries(self, limit: int = 50) -> List[Dict]:
        columns = ["id", "timestamp", "date_str", "name", "model_id",
                   "lr_b64", "sr_b64", "gt_b64", "uncertainty_b64", "ndvi_b64",
                   "metrics_json", "params_json"]
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT {', '.join(columns)}
                FROM history
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
        return [self._row_to_dict(r, columns) for r in rows]

    def get_entry(self, entry_id: str) -> Optional[Dict]:
        columns = ["id", "timestamp", "date_str", "name", "model_id",
                   "lr_b64", "sr_b64", "gt_b64", "uncertainty_b64", "ndvi_b64",
                   "metrics_json", "params_json"]
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT {', '.join(columns)}
                FROM history
                WHERE id = ?
            """, (entry_id,))
            r = cursor.fetchone()
        if not r:
            return None
        return self._row_to_dict(r, columns)

    def delete_entry(self, entry_id: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM history WHERE id = ?", (entry_id,))
            conn.commit()
            return cursor.rowcount > 0

    def clear_history(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM history")
            conn.commit()
            print("History store cleared.")


if __name__ == "__main__":
    store = HistoryStore()
    print(f"Current history entries count: {len(store.list_entries())}")