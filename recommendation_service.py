import sqlite3
from kafka import KafkaConsumer
import json
from similarity_finder import TrackSimilarityFinder

class RecommendationService:
    def __init__(self, db_path="recommendations.db", kafka_topic="new_tracks", n_mfcc=13):
        self.db_path = db_path
        self.kafka_topic = kafka_topic
        self.similarity_finder = TrackSimilarityFinder(n_mfcc=n_mfcc)
        self.consumer = KafkaConsumer(self.kafka_topic, bootstrap_servers="localhost:9092", value_deserializer=lambda m: json.loads(m.decode("utf-8")))
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    track_paths TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tracks (
                    track_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    track_path TEXT,
                    features BLOB
                )
            """)
            conn.commit()
    
    def save_track_features(self, user_id, track_path):
        features = self.similarity_finder.extract_features(track_path)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO tracks (user_id, track_path, features) VALUES (?, ?, ?)", 
                           (user_id, track_path, features.tobytes()))
            conn.commit()
    
    def get_user_tracks(self, user_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT track_path FROM tracks WHERE user_id = ?", (user_id,))
            tracks = [row[0] for row in cursor.fetchall()]
        return tracks
    
    def update_recommendations(self, user_id):
        user_tracks = self.get_user_tracks(user_id)
        if not user_tracks:
            return None
        
        all_tracks = self.get_all_tracks_except_user(user_id)
        recommendations = self.similarity_finder.compare_tracks(user_tracks, all_tracks)
        print(f"Recommendations for {user_id}: {recommendations}")
        return recommendations

    def get_all_tracks_except_user(self, user_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT track_path FROM tracks WHERE user_id != ?", (user_id,))
            tracks = [row[0] for row in cursor.fetchall()]
        return tracks

    def listen_to_kafka(self):
        for message in self.consumer:
            user_id = message.value["user_id"]
            track_path = message.value["track_path"]
            print(f"{user_id}: {track_path}")
            self.save_track_features(user_id, track_path)
            self.update_recommendations(user_id)
