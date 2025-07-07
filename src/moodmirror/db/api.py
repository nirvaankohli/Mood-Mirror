import sqlite3
from sqlite3 import Connection
from pathlib import Path
import sys
from datetime import date, timedelta
from typing import Any, List, Tuple, Union, Dict
from datetime import datetime

def get_data_path(*paths: str) -> Path:
    
    if getattr(sys, 'frozen', False):
        base = Path(sys._MEIPASS)
    else:
        base = Path(__file__).parent.parent

    return base.joinpath(*paths)


class init_db():

    def __init__(self, db = 'db', get_connection: bool = True, init_db: bool = True):

        self.db_path = get_data_path('data', f'{db}.db')

        if get_connection:

            conn = self.get_connection()

        if init_db and get_connection:

            self.create_missing_tables(conn)

        if get_connection:

            conn.close()
        



    def get_connection(self) -> Connection:

        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")

        return conn
    
    def create_missing_tables(self, conn = None):
        
        if conn is None:

            connection = None
            conn = self.get_connection()

        else:

            connection = True

        if connection is None:

            conn = self.get_connection()

        cursor = conn.cursor()
        

        cursor.execute(

                """

                CREATE TABLE IF NOT EXISTS sessions (
                    session_id            INTEGER PRIMARY KEY,
                    day                   TEXT    NOT NULL,
                    start_time            TEXT    NOT NULL, 
                    end_time              TEXT,
                    overall_stress_score  REAL
                )
                
                """
            )

        cursor.execute(

                """

                CREATE TABLE IF NOT EXISTS events (

                    event_id INTEGER PRIMARY KEY,
                    session_id INTEGER,
                    timestamp TEXT NOT NULL,
                    number_in_session INTEGER,
                    emotion TEXT,
                    event_type TEXT,
                    weight REAL,
                    current_stress_score REAL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )

                """
            )
        
        conn.commit()

        if connection is None:

            conn.close()
            


class Sessions():


    def __init__(self):

        self.db = init_db('db')

        self.valid_columns = [

            'session_id', 

            'day', 

            'start_time', 
            
            'end_time', 

            'overall_stress_score'

        ]


    def create_session(self, day = None, start_time = None) -> int:

        # Both start_time and day should be a string isoformat format; 'YYYY-MM-DD HH:MM:SS'

        if not day:

            day = date.today().strftime('%Y-%m-%d')

            
        if not start_time:

            start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


        conn = self.db.get_connection()

        cursor = conn.cursor()

        cursor.execute(

            """

            INSERT INTO sessions (day, start_time)
            VALUES (?, ?);

            """, 
            
            (

                day, 
                
                start_time
            
            )
            
            )
        
        session_id = cursor.lastrowid

        conn.commit()

        conn.close()

        return session_id
    

    def close_session(self, session_id: int, end_time, overall_stress_score: float = None):

        # End_time should be a string isoformat format; 'YYYY-MM-DD HH:MM:SS'

        if not end_time:

            end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        conn = self.db.get_connection()

        cursor = conn.cursor()

        cursor.execute(
            
            """

            UPDATE sessions
            SET end_time = ?,
                overall_stress_score = ?
            WHERE session_id = ?;

            """, 
            
            (
                
                end_time, 
                
                overall_stress_score, 
                
                session_id
             
             )
            
            
            )
        
        conn.commit()

        conn.close

        return "Session closed successfully."
    
    def edit_session_row_by_id(
            
            self, 
            
            session_id: int,
            
            updates: dict
            
            ):
        
        conn = self.db.get_connection()

        valid_columns = self.valid_columns[1:]  # Exclude 'session_id' from valid columns
        
        if not updates:

            raise ValueError("No updates provided.")
        
        clauses = []
        params = []
        
        for column, value in updates.items():

            if column not in valid_columns:

                raise ValueError(f"Invalid column name: {column}. Valid columns are: {valid_columns}")

            clauses.append(f"{column} = ?")
            params.append(value)

        params.append(session_id)

        sql = f"""

            UPDATE sessions
                SET {', '.join(clauses)}
            WHERE session_id = ?;
            
            """
        cursor = conn.cursor()

        cursor.execute(sql, params)

        conn.commit()

        conn.close()

        return "Session updated successfully."

    def open_sessions(self):

        conn = self.db.get_connection()

        cursor = conn.cursor()

        cursor.execute(
            
            """

            SELECT session_id, day, start_time
            FROM sessions
            WHERE end_time IS NULL;


            """
            
            )
        
        ongoing = cursor.fetchall()

        conn.close()

        return ongoing if ongoing else "No ongoing sessions found."
    

    def get_all_sessions(self):

        conn = self.db.get_connection()

        cursor = conn.cursor()

        cursor.execute(
            
            "SELECT * FROM sessions;"
            
            )
        
        all_sessions = cursor.fetchall()

        conn.close()

        return all_sessions if all_sessions else "No sessions found."


    def get_sessions_in_last_x_days(self, days: int):

        conn = self.db.get_connection()

        cutoff = date.today() - timedelta(days=days)

        cursor = conn.cursor()

        cursor.execute(

            "SELECT * FROM sessions WHERE day >= ? ORDER BY day DESC;",

            (cutoff.isoformat(),)

        )

        sessions = cursor.fetchall()

        conn.close()

        return sessions
    

    def query_session_by_column(
            
            self,

            column: str,

            value,

            exact: bool = True
    ):
        
        # if exact is True, it will search for exact matches, otherwise it will search for partial matches (LIKE query).

        
        if column not in self.valid_columns:

            raise ValueError(f"Invalid column name: {column}. Valid columns are: {self.valid_columns}")
        
        conn = self.db.get_connection()

        cursor = conn.cursor()

        if exact:

            sql = f"SELECT * FROM sessions WHERE {column} = ?;"

            params = (value,)

        else:

            sql = f"SELECT * FROM sessions WHERE {column} LIKE ?;"

            params = (f"%{value}%",)

        cursor.execute(
            
            sql, 
            
            params
            
            )
        
        results = cursor.fetchall()
        
        conn.close()

        return results
    

    def query_sessions_by_stress_score(
            
            self,

            threshold: float,

            operator: str = '>=',
    ):
        
        conn = self.db.get_connection()

        if operator not in ['>=', '<=', '=', '>', '<']:

            raise ValueError("Invalid operator. Use one of: '>=', '<=', '=', '>', '<'.")

        cursor = conn.cursor()

        sql = f"SELECT * FROM sessions WHERE overall_stress_score {operator} ?;"

        cursor.execute(
            
            sql, 
            
            (threshold,)
            
            )
        
        results = cursor.fetchall()

        conn.close()
        
        return results
    

class Events():

    def __init__(self):

        self.db = init_db('db')

        self.valid_columns = [

            'event_id', 

            'session_id', 

            'timestamp', 

            'number_in_session', 

            'emotion', 

            'event_type', 

            'weight', 

            'current_stress_score'
        ]
    
    def report_event(
            
            self,

            session_id: int,

            emotion: str,

            event_type: str,

            weight: float,

    ):
        
        conn = self.db.get_connection()

        cursor = conn.cursor()

        cursor.execute(
            
            """

            SELECT
                current_stress_score,
                number_in_session
            FROM events
            WHERE session_id = ?
            ORDER BY number_in_session DESC
            LIMIT 1

            """,

            (session_id,)
        )

        row = cursor.fetchone()

        if row:
            current_stress_score, number_in_session = row
        else:

            current_stress_score = None
            number_in_session = 0
        
        next_number = number_in_session + 1

        stress_score = 0

        if current_stress_score is not None:

            stress_score = ((current_stress_score*number_in_session) + ((weight * 10)))/next_number

        else:

            stress_score = weight * 10

        ts = date.today().isoformat() + " " + date.today().strftime("%H:%M:%S")

        cursor.execute(

            """

            INSERT INTO events
            (session_id, timestamp, number_in_session, emotion, event_type, weight, current_stress_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            
            """,

        (

            session_id, 
            ts, 
            next_number, 
            emotion, 
            event_type, 
            weight, 
            stress_score

        )

        )

        conn.commit()

        conn.close()

        return cursor.lastrowid
    
    def query_by_column(
            
            self,
    
            column: str,
    
            value: Union[str, int, float],
    
            exact: bool = True,
    
            epsilon: float = 0.01,
    
    ):
        
        conn = self.db.get_connection()
        cursor = conn.cursor()
    
        allowed = {
            'event_id', 'session_id', 'timestamp', 'number_in_session',
            'emotion', 'event_type', 'weight', 'current_stress_score'
        }
        if column not in allowed:
            raise ValueError(f"Invalid column: {column}")
    
        if exact:
            sql = f"""
                SELECT *
                  FROM events
                 WHERE {column} = ?
            """
            params = (value,)
        else:
            if isinstance(value, (int, float)):
                sql = f"""
                    SELECT *
                      FROM events
                     WHERE ABS({column} - ?) <= ?
                """
                params = (value, epsilon)
            else:
                sql = f"""
                    SELECT *
                      FROM events
                     WHERE {column} LIKE ?
                """
                params = (f"%{value}%",)
    
        cursor.execute(
            
            sql,
    
            params
    
        )
        rows = cursor.fetchall()

        conn.close()

        return rows
    
    
    def get_max_number_in_session(
                
                self,
        
                session_id: int,
        
        ):
            
            conn = self.db.get_connection()
            cursor = conn.cursor()
        
            cursor.execute(
                
                """
                SELECT *
                FROM events
                WHERE session_id = ?
                ORDER BY number_in_session DESC
                LIMIT 1
                """,
        
                (
                    session_id,
                )
        
            )
            result = cursor.fetchone()
            conn.close()
            return result
        
        
    def fetch_events_between_timestamps(
                
                self,
        
                start_ts: str,
        
                end_ts: str,
        
        ):
            
            conn = self.db.get_connection()
            cursor = conn.cursor()
        
            cursor.execute(
                
                """
                SELECT *
                FROM events
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
                """,
        
                (
                    start_ts,
                    end_ts,
                )
        
            )
            rows = cursor.fetchall()

            conn.close()

            return rows
        
        
    def edit_event_by_id(
                
                self,
        
                event_id: int,
        
                updates: Dict[str, Any],
        
        ):
            
            conn = self.db.get_connection()
            cursor = conn.cursor()
        
            allowed = {
                'session_id', 'timestamp', 'number_in_session', 'emotion',
                'event_type', 'weight', 'current_stress_score'
            }

            cols = []
            params: List[Any] = []
            for col, val in updates.items():
                if col not in allowed:
                    raise ValueError(f"Cannot update column: {col}")
                cols.append(f"{col} = ?")
                params.append(val)
        
            if not cols:
                conn.close()
                return
        
            sql = f"UPDATE events SET {', '.join(cols)} WHERE event_id = ?"
            params.append(event_id)
        
            cursor.execute(
                
                sql,
        
                tuple(params)
        
            )

            conn.commit()

            conn.close()
        
        
    def retrieve_all_events(
                
                self,
        
        ):
            
            conn = self.db.get_connection()
            cursor = conn.cursor()
        
            cursor.execute(
                
                """
                SELECT *
                FROM events
                """
        
            )
            rows = cursor.fetchall()

            conn.close()

            return rows
        
        
    def query_with_operator(
                
                self,
        
                column: str,
        
                operator: str,
        
                value: Union[int, float],
        
        ):
            
            conn = self.db.get_connection()
            cursor = conn.cursor()
        
            allowed_cols = {
                'event_id', 'session_id', 'number_in_session',
                'weight', 'current_stress_score'
            }
            allowed_ops = {'=', '!=', '<', '<=', '>', '>='}
        
            if column not in allowed_cols:
                raise ValueError(f"Invalid column: {column}")
            if operator not in allowed_ops:
                raise ValueError(f"Invalid operator: {operator}")
        
            sql = f"SELECT * FROM events WHERE {column} {operator} ?"
        
            cursor.execute(
                
                sql,
        
                (value,)
        
            )

            rows = cursor.fetchall()

            conn.close()

            return rows

if __name__ == "__main__":

    from pprint import pprint

    sessions = Sessions()
    events = Events()
    
    pprint(sessions.get_all_sessions())
    pprint(events.retrieve_all_events())
    