"""
Chargement et utilisation des modèles de recommandation - Version Finale Corrigée
"""

import pickle
import json
import numpy as np
import pandas as pd
from scipy import sparse
from typing import List, Tuple, Dict, Optional
import streamlit as st
import os

from config import *


class RecommendationSystem:
    """Classe principale pour gérer les trois modèles de recommandation"""
    
    def __init__(self):
        """Initialisation du système de recommandation"""
        self.content_model = None
        self.collaborative_model = None
        self.hybrid_config = None
        self.songs_metadata = None
        self.train_df = None
        self.id_to_idx = None
        self.idx_to_id = None
        self.topN_by_user = None
        
    @st.cache_resource
    def load_all(_self):
        """Charge tous les modèles et données nécessaires"""
        try:
            # ========================================
            # 1. CHARGER songs_content_features.csv (CONTIENT TOUT)
            # ========================================
            try:
                _self.songs_metadata = pd.read_csv(SONGS_CONTENT_FEATURES)
                
                # Vérifier les colonnes essentielles
                required_cols = ['song_id', 'title', 'artist', 'genre']
                missing_cols = [col for col in required_cols if col not in _self.songs_metadata.columns]
                
                if missing_cols:
                    st.error(f"❌ Colonnes manquantes : {missing_cols}")
                    return False
                
                # Lister les features audio disponibles
                audio_cols = ['tempo', 'energy', 'danceability', 'valence', 'acousticness', 
                              'instrumentalness', 'liveness', 'loudness', 'speechiness', 'key', 
                              'mode', 'time_signature']
                
                available_audio = [col for col in audio_cols if col in _self.songs_metadata.columns]
                
                # Remplir les valeurs manquantes par 0 (sauf pour mode qui est catégoriel)
                for col in available_audio:
                    if col != 'mode':  # mode est catégoriel (Major/Minor)
                        _self.songs_metadata[col] = pd.to_numeric(_self.songs_metadata[col], errors='coerce').fillna(0)
                    else:
                        _self.songs_metadata[col] = _self.songs_metadata[col].fillna('Unknown')
                
            except FileNotFoundError:
                st.error(f"❌ Fichier non trouvé : {SONGS_CONTENT_FEATURES}")
                st.error("⚠️ Exécutez d'abord le Notebook 2 (cellule de création songs_content_features.csv) !")
                return False
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement des métadonnées : {e}")
                import traceback
                st.error(traceback.format_exc())
                return False
            
            # ========================================
            # 2. CHARGER LES DONNÉES D'ENTRAÎNEMENT
            # ========================================
            try:
                _self.train_df = pd.read_csv(TRAIN_DATA)
            except Exception as e:
                st.error(f"❌ Impossible de charger train_data.csv: {e}")
                return False
            
            # ========================================
            # 3. CRÉER LES MAPPINGS
            # ========================================
            _self.id_to_idx = {song_id: i for i, song_id in enumerate(_self.songs_metadata['song_id'])}
            _self.idx_to_id = {i: song_id for song_id, i in _self.id_to_idx.items()}
            
            # ========================================
            # 4. CHARGER LE MODÈLE CONTENT-BASED
            # ========================================
            try:
                with open(CONTENT_MODEL, 'rb') as f:
                    content_data = pickle.load(f)
                
                # Vérifier le format
                if isinstance(content_data, dict):
                    _self.content_model = content_data.get('similarity_matrix')
                    if 'id_to_idx' in content_data:
                        _self.id_to_idx = content_data['id_to_idx']
                    if 'idx_to_id' in content_data:
                        _self.idx_to_id = content_data['idx_to_id']
                else:
                    # Si c'est juste la matrice
                    _self.content_model = content_data
                
            except FileNotFoundError:
                st.error(f"❌ Fichier non trouvé : {CONTENT_MODEL}")
                st.error("⚠️ Exécutez d'abord le Notebook 3 (Cellule 1 - Content-Based) !")
                _self.content_model = None
            except Exception as e:
                st.error(f"❌ Erreur Content-Based: {str(e)}")
                _self.content_model = None
            
            # ========================================
            # 5. CHARGER LE MODÈLE COLLABORATIVE
            # ========================================
            try:
                with open(COLLABORATIVE_MODEL, 'rb') as f:
                    collab_data = pickle.load(f)
                
                if isinstance(collab_data, dict):
                    _self.topN_by_user = collab_data.get('topN_by_user', {})
                else:
                    _self.topN_by_user = {}
                
            except FileNotFoundError:
                st.error(f"❌ Fichier non trouvé : {COLLABORATIVE_MODEL}")
                st.error("⚠️ Exécutez d'abord le Notebook 3 (Cellule 2 - Collaborative) !")
                _self.topN_by_user = {}
            except Exception as e:
                st.error(f"❌ Erreur Collaborative: {str(e)}")
                _self.topN_by_user = {}
            
            # ========================================
            # 6. CHARGER LA CONFIGURATION HYBRID
            # ========================================
            try:
                with open(HYBRID_CONFIG, 'r') as f:
                    _self.hybrid_config = json.load(f)
            except FileNotFoundError:
                _self.hybrid_config = {
                    'content_weight': 0.5,
                    'collaborative_weight': 0.5
                }
            except Exception as e:
                _self.hybrid_config = {
                    'content_weight': 0.5,
                    'collaborative_weight': 0.5
                }
            
            return True
            
        except Exception as e:
            st.error(f"❌ Erreur globale lors du chargement : {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return False
    
    def get_song_info(self, song_id: int) -> Dict:
        """
        Récupère les informations complètes d'une chanson avec TOUTES les features audio
        
        Args:
            song_id: ID de la chanson
            
        Returns:
            Dictionnaire avec toutes les informations de la chanson
        """
        song = self.songs_metadata[self.songs_metadata['song_id'] == song_id]
        
        if song.empty:
            # Retourner un dictionnaire avec valeurs par défaut si chanson non trouvée
            return {
                'song_id': song_id,
                'title': f'Song {song_id}',
                'artist': 'Unknown',
                'genre': 'Unknown',
                'album': 'Unknown',
                'release_year': 'N/A',
                'duration_sec': 0,
                'popularity': 0,
                'language': 'Unknown',
                'explicit': 'No',
                'tempo': 0,
                'energy': 0,
                'danceability': 0,
                'valence': 0,
                'acousticness': 0,
                'instrumentalness': 0,
                'liveness': 0,
                'loudness': 0,
                'speechiness': 0,
                'key': 0,
                'mode': 'Unknown',
                'time_signature': 0
            }
        
        song_data = song.iloc[0]
        
        # Construire le dictionnaire avec TOUTES les métadonnées
        song_info = {
            'song_id': int(song_data['song_id']),
            'title': str(song_data.get('title', f'Song {song_id}')),
            'artist': str(song_data.get('artist', 'Unknown')),
            'album': str(song_data.get('album', 'Unknown')),
            'genre': str(song_data.get('genre', 'Unknown')),
            'release_year': song_data.get('release_year', 'N/A'),
            'duration_sec': float(song_data.get('duration_sec', 0)),
            'popularity': float(song_data.get('popularity', 0)),
            'language': str(song_data.get('language', 'Unknown')),
            'explicit': str(song_data.get('explicit', 'No'))
        }
        
        # Ajouter TOUTES les features audio disponibles
        audio_features = {
            'tempo': 'tempo',
            'energy': 'energy',
            'danceability': 'danceability',
            'valence': 'valence',
            'acousticness': 'acousticness',
            'instrumentalness': 'instrumentalness',
            'liveness': 'liveness',
            'loudness': 'loudness',
            'speechiness': 'speechiness',
            'key': 'key',
            'mode': 'mode',
            'time_signature': 'time_signature'
        }
        
        for key, col in audio_features.items():
            if col in song_data.index:
                value = song_data[col]
                
                # Convertir en float si numérique, sinon garder tel quel
                if col != 'mode':  # mode est catégoriel (Major/Minor)
                    try:
                        song_info[key] = float(value) if pd.notna(value) else 0.0
                    except (ValueError, TypeError):
                        song_info[key] = 0.0
                else:
                    # Pour mode, garder comme string
                    song_info[key] = str(value) if pd.notna(value) and str(value) != 'nan' else 'Unknown'
            else:
                # Valeur par défaut si la colonne n'existe pas
                song_info[key] = 'Unknown' if col == 'mode' else 0.0
        
        return song_info
    
    def get_user_history(self, user_id: int, n: int = 10) -> List[Dict]:
        """Récupère l'historique d'écoute d'un utilisateur"""
        user_data = self.train_df[self.train_df['user_id'] == user_id]
        
        if user_data.empty:
            return []
        
        # Trier par timestamp décroissant
        user_data = user_data.sort_values('timestamp', ascending=False).head(n)
        
        history = []
        for _, row in user_data.iterrows():
            song_info = self.get_song_info(row['song_id'])
            song_info['liked'] = int(row.get('liked', 0))
            song_info['timestamp'] = row.get('timestamp', '')
            history.append(song_info)
        
        return history
    
    def similar_items(self, seed_song_id: int, topk: int = 10) -> List[Tuple[int, float]]:
        """Recommandation Content-Based : chansons similaires à une seed"""
        
        if self.content_model is None:
            return []
        
        idx = self.id_to_idx.get(seed_song_id)
        if idx is None:
            return []
        
        try:
            # Récupérer la ligne de similarité
            similarities = self.content_model[idx]
            
            # Si matrice sparse
            if sparse.issparse(similarities):
                similarities = similarities.toarray().ravel()
            
            # Exclure la chanson seed
            similarities[idx] = -1.0
            
            # Top-K indices
            top_indices = similarities.argsort()[::-1][:topk]
            
            # Retourner (song_id, score)
            return [(self.idx_to_id[i], float(similarities[i])) for i in top_indices]
        
        except Exception as e:
            st.error(f"Erreur similar_items: {e}")
            return []
    
    def recommend_content_based(self, user_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """Recommandation Content-Based pour un utilisateur"""
        # Récupérer l'historique
        history = self.train_df[self.train_df['user_id'] == user_id]
        
        if history.empty:
            return []
        
        # Dernière chanson écoutée
        seed_song = history.sort_values('timestamp').iloc[-1]['song_id']
        
        # Recommandations similaires
        return self.similar_items(seed_song, topk=n)
    
    def recommend_collaborative(self, user_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """Recommandation Collaborative Filtering (SVD)"""
        uid = str(user_id)
        
        if uid not in self.topN_by_user:
            return []  # Cold start
        
        # Récupérer les recommandations pré-calculées
        return self.topN_by_user[uid][:n]
    
    def _normalize_scores(self, pairs: List[Tuple[int, float]]) -> Dict[int, float]:
        """Normalise les scores dans [0, 1]"""
        if not pairs:
            return {}
        
        scores = np.array([s for _, s in pairs], dtype=float)
        
        # Gérer les NaN
        if np.isnan(scores).any():
            scores = np.nan_to_num(scores, nan=0.0)
        
        # Min-Max normalization
        if scores.ptp() > 0:
            scores = (scores - scores.min()) / scores.ptp()
        else:
            scores = np.ones_like(scores)
        
        return {song_id: float(score) for (song_id, _), score in zip(pairs, scores)}
    
    def recommend_hybrid(self, user_id: int, n: int = 10, 
                        alpha: Optional[float] = None, 
                        beta: Optional[float] = None) -> List[Tuple[int, float]]:
        """Recommandation Hybride avec fallback intelligent"""
        
        # Utiliser les poids par défaut si non spécifiés
        if alpha is None:
            alpha = self.hybrid_config.get('content_weight', DEFAULT_ALPHA)
        if beta is None:
            beta = self.hybrid_config.get('collaborative_weight', DEFAULT_BETA)
        
        # Obtenir les recommandations de chaque modèle
        try:
            content_recs = self.recommend_content_based(user_id, n=n*2)
        except:
            content_recs = []
        
        try:
            collab_recs = self.recommend_collaborative(user_id, n=n*2)
        except:
            collab_recs = []
        
        # STRATÉGIE DE FALLBACK
        if not content_recs and not collab_recs:
            return []
        
        if content_recs and not collab_recs:
            return content_recs[:n]  # Fallback Content
        
        if collab_recs and not content_recs:
            return collab_recs[:n]   # Fallback Collaborative
        
        # FUSION PONDÉRÉE (les deux disponibles)
        C = self._normalize_scores(content_recs)
        CF = self._normalize_scores(collab_recs)
        
        all_songs = set(C.keys()) | set(CF.keys())
        
        hybrid_scores = {
            song_id: alpha * C.get(song_id, 0) + beta * CF.get(song_id, 0)
            for song_id in all_songs
        }
        
        # Trier par score décroissant
        ranked = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        
        return ranked[:n]
    
    def get_recommendations_all_models(self, user_id: int, n: int = 10) -> Dict[str, List[Dict]]:
        """Génère des recommandations avec les 3 modèles"""
        results = {
            'content': [],
            'collaborative': [],
            'hybrid': []
        }
        
        # Content-Based
        content_recs = self.recommend_content_based(user_id, n)
        for song_id, score in content_recs:
            song_info = self.get_song_info(song_id)
            song_info['score'] = score
            results['content'].append(song_info)
        
        # Collaborative
        collab_recs = self.recommend_collaborative(user_id, n)
        for song_id, score in collab_recs:
            song_info = self.get_song_info(song_id)
            song_info['score'] = score
            results['collaborative'].append(song_info)
        
        # Hybrid
        hybrid_recs = self.recommend_hybrid(user_id, n)
        for song_id, score in hybrid_recs:
            song_info = self.get_song_info(song_id)
            song_info['score'] = score
            results['hybrid'].append(song_info)
        
        return results
    
    def get_model_stats(self) -> Dict:
        """Récupère les statistiques des modèles"""
        try:
            with open(MODELING_SUMMARY, 'r') as f:
                summary = json.load(f)
            return summary
        except:
            return {
                'dataset': {
                    'n_users': len(self.train_df['user_id'].unique()) if self.train_df is not None else 0,
                    'n_songs': len(self.songs_metadata) if self.songs_metadata is not None else 0,
                    'n_interactions_train': len(self.train_df) if self.train_df is not None else 0
                }
            }