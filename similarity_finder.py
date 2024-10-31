import librosa
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

class TrackSimilarityFinder:
    def __init__(self, n_mfcc=13):
        self.n_mfcc = n_mfcc
    
    def extract_features(self, track_path):
        y, sr = librosa.load(track_path, sr=None)
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        # Хрома-фичи
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Спектральный контраст
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)
        
        features = np.hstack([mfcc_mean, chroma_mean, contrast_mean])
        print(features)
        return features
    
    def compare_tracks(self, reference_tracks, target_tracks):
        reference_features = [self.extract_features(track) for track in reference_tracks]
        
        target_features = [self.extract_features(track) for track in target_tracks]
        
        similarity_dict = {}
        
        for i, ref_feat in enumerate(reference_features):
            similarities = cosine_similarity([ref_feat], target_features).flatten()
            
            scored_similarities = [(target_tracks[j], similarities[j]) for j in range(len(target_tracks))]

            scored_similarities.sort(key=lambda x: x[1], reverse=True)
            
            similarity_dict[reference_tracks[i]] = scored_similarities
        
        return similarity_dict
    def find_best_matches(self, reference_tracks, target_tracks):
        similarity_scores = defaultdict(float)

        for ref_track in reference_tracks:
            similarities = self.compare_tracks([ref_track], target_tracks)
            for _, score in similarities[ref_track]:
                similarity_scores[ref_track] += score

        sorted_matches = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_matches
    

reference_tracks = ["tracks/ПошлаяМолли.mp3", "tracks/Bearwolf.mp3", "tracks/Whyspurky.mp3"]
target_tracks = ["tracks/Blizkey.mp3", "tracks/Navai.mp3", "tracks/Linkin.mp3"]

finder = TrackSimilarityFinder()
similar_tracks_with_scores = finder.find_best_matches(reference_tracks, target_tracks)

for track, score in similar_tracks_with_scores:
    print(f"{track} - {score:.4f}")