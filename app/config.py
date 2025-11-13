"""
Configuration de l'application Streamlit
"""

import os

# Chemins des fichiers
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# Fichiers de données
TRAIN_DATA = os.path.join(PROCESSED_DIR, 'train_data.csv')
TEST_DATA = os.path.join(PROCESSED_DIR, 'test_data.csv')
SONGS_METADATA = os.path.join(PROCESSED_DIR, 'songs_metadata.csv')
SONGS_CONTENT_FEATURES = os.path.join(PROCESSED_DIR, 'songs_content_features.csv')

# Fichiers de modèles
CONTENT_MODEL = os.path.join(MODELS_DIR, 'content_based_model.pkl')
COLLABORATIVE_MODEL = os.path.join(MODELS_DIR, 'collaborative_model.pkl')
HYBRID_CONFIG = os.path.join(MODELS_DIR, 'hybrid_config.json')
EVALUATION_REPORT = os.path.join(MODELS_DIR, 'evaluation_report.json')
MODELING_SUMMARY = os.path.join(MODELS_DIR, 'modeling_summary.json')

# Configuration de l'app
APP_TITLE = "🎵 Système de Recommandation Musicale"
APP_ICON = "🎵"
LAYOUT = "wide"
SIDEBAR_STATE = "expanded"

# Paramètres de recommandation
DEFAULT_N_RECOMMENDATIONS = 10
MAX_RECOMMENDATIONS = 20
MIN_RECOMMENDATIONS = 5

# Paramètres du modèle Hybrid
DEFAULT_ALPHA = 0.5  # Poids Content-Based
DEFAULT_BETA = 0.5   # Poids Collaborative

# Styles CSS personnalisés
CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1DB954;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .song-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1DB954;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .recommendation-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #333;
        margin: 1rem 0;
    }
</style>
"""

# Messages
MSG_LOADING = "⏳ Chargement des données..."
MSG_MODEL_LOADED = "✅ Modèles chargés avec succès !"
MSG_ERROR_LOADING = "❌ Erreur lors du chargement"
MSG_NO_HISTORY = "⚠️ Aucun historique disponible pour cet utilisateur"
MSG_GENERATING_RECS = "🎵 Génération des recommandations..."

# Couleurs
COLOR_PRIMARY = "#1DB954"  # Vert Spotify
COLOR_SECONDARY = "#191414"  # Noir Spotify
COLOR_ACCENT = "#FF6B6B"
COLOR_SUCCESS = "#4ECDC4"

# Emojis par genre
GENRE_EMOJIS = {
    'Pop': '🎤',
    'Rock': '🎸',
    'Jazz': '🎷',
    'EDM': '🎧',
    'Classical': '🎻',
    'Hip Hop': '🎤',
    'Country': '🤠',
    'R&B': '🎵',
    'Blues': '🎺',
    'Metal': '🤘',
    'Indie': '🎸',
    'Folk': '🪕',
    'Reggae': '🌴',
    'Latin': '💃',
    'Electronic': '🔊'
}

# Emojis par contexte
CONTEXT_EMOJIS = {
    'Workout': '💪',
    'Relax': '😌',
    'Commute': '🚗',
    'Focus': '🎯',
    'Party': '🎉',
    'Sleep': '😴',
    'Study': '📚'
}

# Emojis par émotion
EMOTION_EMOJIS = {
    'Happy': '😊',
    'Sad': '😢',
    'Calm': '😌',
    'Energetic': '⚡',
    'Romantic': '❤️',
    'Angry': '😠',
    'Nostalgic': '🌅'
}