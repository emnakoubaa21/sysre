"""
Fonctions utilitaires pour l'affichage dans Streamlit - Version Finale Corrigée
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict
import numpy as np


def display_song_card(song: Dict, show_score: bool = True, show_details: bool = True):
    """
    Affiche une carte de chanson avec ses informations complètes
    
    Args:
        song: Dictionnaire contenant les infos de la chanson
        show_score: Afficher le score de recommandation
        show_details: Afficher les détails audio
    """
    with st.container():
        # Titre de la chanson
        st.markdown(f"### 🎵 {song.get('title', 'Unknown')}")
        
        # Informations principales sur 3 colonnes
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"**Artiste:** {song.get('artist', 'Unknown')}")
            st.write(f"**Genre:** {song.get('genre', 'Unknown')}")
        
        with col2:
            st.write(f"**Album:** {song.get('album', 'N/A')}")
            st.write(f"**Année:** {song.get('release_year', 'N/A')}")
        
        with col3:
            # Durée
            duration = song.get('duration_sec', 0)
            if duration > 0:
                minutes = int(duration // 60)
                seconds = int(duration % 60)
                st.write(f"**Durée:** {minutes}:{seconds:02d}")
            else:
                st.write(f"**Durée:** N/A")
            
            # Popularité
            popularity = song.get('popularity', 0)
            if popularity > 0:
                st.write(f"**Popularité:** {int(popularity)}/100")
        
        # Afficher le score si demandé
        if show_score and 'score' in song:
            score = song['score']
            
            # ✅ CORRECTION : Vérifier et normaliser le score
            if score is not None:
                try:
                    score = float(score)
                    
                    # Si le score est > 1, le normaliser
                    if score > 1.0:
                        score = min(score / 10.0, 1.0)
                    
                    # S'assurer que le score est entre 0 et 1
                    score = max(0.0, min(1.0, abs(score)))
                    
                    st.progress(score)
                    st.caption(f"**Score:** {score:.3f}")
                    
                except (ValueError, TypeError):
                    st.caption(f"Score: N/A")
        
        # ✅ CORRECTION : Afficher les détails audio avec dénormalisation intelligente
        if show_details:
            with st.expander("🎧 Détails audio"):
                # Vérifier si les features audio existent et ne sont pas toutes à 0
                has_audio_features = any([
                    song.get('tempo', 0) != 0,
                    song.get('energy', 0) != 0,
                    song.get('danceability', 0) != 0,
                    song.get('valence', 0) != 0
                ])
                
                if not has_audio_features:
                    st.warning("⚠️ Pas de features audio disponibles pour cette chanson")
                else:
                    # ✅ COLONNES PRINCIPALES
                    detail_col1, detail_col2 = st.columns(2)
                    
                    with detail_col1:
                        # Tempo (dénormaliser si normalisé)
                        tempo = song.get('tempo', 0)
                        if tempo != 0:
                            # Si tempo est normalisé (entre -1 et 1), le dénormaliser
                            if abs(tempo) <= 2:  # Probablement normalisé
                                # Supposons moyenne=120 BPM, std=30 BPM
                                tempo_display = tempo * 30 + 120
                            else:
                                tempo_display = tempo
                            st.metric("🥁 Tempo", f"{tempo_display:.0f} BPM")
                        else:
                            st.metric("🥁 Tempo", "N/A")
                        
                        # Energy (dénormaliser si normalisé)
                        energy = song.get('energy', 0)
                        if energy != 0:
                            # Si normalisé entre [-1, 1], convertir vers [0, 1]
                            if abs(energy) <= 2:
                                energy_display = (energy + 1) / 2
                            else:
                                energy_display = energy
                            
                            # S'assurer que c'est entre 0 et 1
                            energy_display = max(0.0, min(1.0, energy_display))
                            st.metric("⚡ Energy", f"{energy_display:.2f}")
                        else:
                            st.metric("⚡ Energy", "N/A")
                    
                    with detail_col2:
                        # Danceability (dénormaliser si normalisé)
                        danceability = song.get('danceability', 0)
                        if danceability != 0:
                            if abs(danceability) <= 2:
                                danceability_display = (danceability + 1) / 2
                            else:
                                danceability_display = danceability
                            
                            danceability_display = max(0.0, min(1.0, danceability_display))
                            st.metric("💃 Danceability", f"{danceability_display:.2f}")
                        else:
                            st.metric("💃 Danceability", "N/A")
                        
                        # Valence (dénormaliser si normalisé)
                        valence = song.get('valence', 0)
                        if valence != 0:
                            if abs(valence) <= 2:
                                valence_display = (valence + 1) / 2
                            else:
                                valence_display = valence
                            
                            valence_display = max(0.0, min(1.0, valence_display))
                            st.metric("😊 Valence", f"{valence_display:.2f}")
                        else:
                            st.metric("😊 Valence", "N/A")
                    
                    # ✅ SÉPARATEUR
                    st.markdown("---")
                    
                    # ✅ INFORMATIONS SUPPLÉMENTAIRES
                    extra_col1, extra_col2 = st.columns(2)
                    
                    with extra_col1:
                        # Acousticness
                        acousticness = song.get('acousticness', 0)
                        if acousticness != 0:
                            if abs(acousticness) <= 2:
                                acousticness_display = (acousticness + 1) / 2
                            else:
                                acousticness_display = acousticness
                            
                            acousticness_display = max(0.0, min(1.0, acousticness_display))
                            st.caption(f"🎸 **Acoustique:** {acousticness_display:.2f}")
                        
                        # Instrumentalness
                        instrumentalness = song.get('instrumentalness', 0)
                        if instrumentalness != 0:
                            if abs(instrumentalness) <= 2:
                                instrumentalness_display = (instrumentalness + 1) / 2
                            else:
                                instrumentalness_display = instrumentalness
                            
                            instrumentalness_display = max(0.0, min(1.0, instrumentalness_display))
                            st.caption(f"🎼 **Instrumental:** {instrumentalness_display:.2f}")
                        
                        # Liveness
                        liveness = song.get('liveness', 0)
                        if liveness != 0:
                            if abs(liveness) <= 2:
                                liveness_display = (liveness + 1) / 2
                            else:
                                liveness_display = liveness
                            
                            liveness_display = max(0.0, min(1.0, liveness_display))
                            st.caption(f"🎤 **Live:** {liveness_display:.2f}")
                    
                    with extra_col2:
                        # Mode
                        mode = song.get('mode', 'Unknown')
                        if mode not in ['Unknown', None, 0, '0']:
                            st.caption(f"🎹 **Mode:** {mode}")
                        
                        # Key (Tonalité)
                        key = song.get('key', 0)
                        if key != 0:
                            # Mapping des clés musicales
                            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                            key_idx = int(key) % 12
                            st.caption(f"🎼 **Tonalité:** {keys[key_idx]}")
                        
                        # Loudness
                        loudness = song.get('loudness', 0)
                        if loudness != 0:
                            st.caption(f"🔊 **Volume:** {loudness:.2f} dB")
                    
                    # ✅ SECTION ADDITIONNELLE
                    st.markdown("---")
                    
                    additional_col1, additional_col2 = st.columns(2)
                    
                    with additional_col1:
                        # Language
                        language = song.get('language', 'Unknown')
                        if language not in ['Unknown', None]:
                            st.caption(f"🌍 **Langue:** {language}")
                    
                    with additional_col2:
                        # Explicit
                        explicit = song.get('explicit', 'No')
                        if explicit == 'Yes':
                            st.caption(f"🔞 **Explicite:** Oui")
        
        st.markdown("---")


def display_recommendations_table(recommendations: List[Dict]):
    """
    Affiche les recommandations sous forme de tableau
    
    Args:
        recommendations: Liste de dictionnaires contenant les infos des chansons
    """
    if not recommendations:
        st.warning("Aucune recommandation à afficher")
        return
    
    # Créer le DataFrame
    df = pd.DataFrame(recommendations)
    
    # Sélectionner les colonnes à afficher
    columns_to_display = ['title', 'artist', 'genre', 'album', 'release_year']
    
    # Ajouter le score si disponible
    if 'score' in df.columns:
        # ✅ CORRECTION : Normaliser les scores pour l'affichage
        def normalize_score(x):
            try:
                score = float(x)
                if score > 1.0:
                    score = min(score / 10.0, 1.0)
                return max(0.0, min(1.0, abs(score)))
            except:
                return 0.0
        
        df['score_display'] = df['score'].apply(normalize_score)
        df['score_display'] = df['score_display'].apply(lambda x: f"{x:.3f}")
        columns_to_display.append('score_display')
    
    # Filtrer les colonnes qui existent
    available_columns = [col for col in columns_to_display if col in df.columns]
    
    if not available_columns:
        st.error("Aucune colonne valide trouvée dans les recommandations")
        return
    
    df_display = df[available_columns].copy()
    
    # Renommer les colonnes pour l'affichage
    column_names = {
        'title': 'Titre',
        'artist': 'Artiste',
        'genre': 'Genre',
        'album': 'Album',
        'release_year': 'Année',
        'score_display': 'Score'
    }
    
    df_display = df_display.rename(columns={
        col: column_names.get(col, col) for col in df_display.columns
    })
    
    # Afficher le tableau
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True
    )


def plot_genre_distribution(recommendations: Dict[str, List[Dict]]):
    """
    Affiche la distribution des genres par modèle
    
    Args:
        recommendations: Dict avec clés 'content', 'collaborative', 'hybrid'
    """
    genre_counts = {}
    
    for model_name, recs in recommendations.items():
        if recs:
            df = pd.DataFrame(recs)
            if 'genre' in df.columns:
                genre_counts[model_name] = df['genre'].value_counts().to_dict()
            else:
                genre_counts[model_name] = {}
        else:
            genre_counts[model_name] = {}
    
    # Créer un DataFrame pour Plotly
    all_genres = set()
    for counts in genre_counts.values():
        all_genres.update(counts.keys())
    
    if not all_genres:
        st.info("Pas de données de genre disponibles pour le graphique")
        return
    
    data = []
    for genre in all_genres:
        data.append({
            'Genre': genre,
            'Content-Based': genre_counts.get('content', {}).get(genre, 0),
            'Collaborative': genre_counts.get('collaborative', {}).get(genre, 0),
            'Hybrid': genre_counts.get('hybrid', {}).get(genre, 0)
        })
    
    df_plot = pd.DataFrame(data)
    
    # Créer le graphique
    fig = px.bar(
        df_plot,
        x='Genre',
        y=['Content-Based', 'Collaborative', 'Hybrid'],
        title="Distribution des genres par modèle",
        barmode='group',
        color_discrete_map={
            'Content-Based': '#FF6B6B',
            'Collaborative': '#4ECDC4',
            'Hybrid': '#45B7D1'
        }
    )
    
    fig.update_layout(
        xaxis_title="Genre",
        yaxis_title="Nombre de chansons",
        legend_title="Modèle",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_diversity_comparison(recommendations: Dict[str, List[Dict]]):
    """
    Compare la diversité des recommandations entre modèles
    
    Args:
        recommendations: Dict avec clés 'content', 'collaborative', 'hybrid'
    """
    diversity_scores = {}
    
    for model_name, recs in recommendations.items():
        if recs:
            df = pd.DataFrame(recs)
            if 'genre' in df.columns:
                unique_genres = df['genre'].nunique()
                total_recs = len(df)
                diversity = unique_genres / total_recs if total_recs > 0 else 0
                diversity_scores[model_name] = diversity
            else:
                diversity_scores[model_name] = 0
        else:
            diversity_scores[model_name] = 0
    
    if not diversity_scores or all(v == 0 for v in diversity_scores.values()):
        st.info("Pas assez de données pour calculer la diversité")
        return
    
    # Créer le graphique
    fig = go.Figure(data=[
        go.Bar(
            x=list(diversity_scores.keys()),
            y=list(diversity_scores.values()),
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
            text=[f"{v:.2%}" for v in diversity_scores.values()],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Diversité des recommandations (genres uniques / total)",
        xaxis_title="Modèle",
        yaxis_title="Score de diversité",
        height=400,
        yaxis=dict(tickformat='.0%')
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_user_history(history: List[Dict], n: int = 10):
    """
    Affiche l'historique d'écoute d'un utilisateur
    
    Args:
        history: Liste des chansons écoutées
        n: Nombre de chansons à afficher
    """
    if not history:
        st.info("Aucun historique disponible pour cet utilisateur")
        return
    
    st.subheader(f"📜 Historique d'écoute ({len(history)} interactions)")
    
    for i, song in enumerate(history[:n], 1):
        with st.container():
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.write(f"**{i}. {song.get('title', 'Unknown')}**")
                st.caption(f"Artiste: {song.get('artist', 'Unknown')}")
            
            with col2:
                st.write(f"Genre: {song.get('genre', 'Unknown')}")
                st.caption(f"Album: {song.get('album', 'N/A')}")
            
            with col3:
                liked = song.get('liked', 0)
                if liked == 1:
                    st.write("❤️ Aimé")
                else:
                    st.write("👍 Écouté")
            
            st.markdown("---")


def format_model_name(model_key: str) -> str:
    """
    Formate le nom du modèle pour l'affichage
    
    Args:
        model_key: Clé du modèle ('content', 'collaborative', 'hybrid')
    
    Returns:
        Nom formaté du modèle
    """
    names = {
        'content': 'Content-Based',
        'collaborative': 'Collaborative',
        'hybrid': 'Hybrid'
    }
    return names.get(model_key, model_key.title())


def display_audio_features_chart(song: Dict):
    """
    Affiche un graphique radar des features audio d'une chanson
    
    Args:
        song: Dictionnaire contenant les infos de la chanson
    """
    # Features à afficher
    features = ['energy', 'danceability', 'valence', 'acousticness', 'instrumentalness', 'liveness']
    
    # Récupérer les valeurs et dénormaliser
    values = []
    labels = []
    
    for feature in features:
        if feature in song and song[feature] != 0:
            value = song[feature]
            
            # Dénormaliser si nécessaire
            if abs(value) <= 2:
                value = (value + 1) / 2
            
            value = max(0.0, min(1.0, value))
            values.append(value)
            labels.append(feature.capitalize())
    
    if not values:
        st.info("Pas assez de features audio pour créer le graphique")
        return
    
    # Créer le graphique radar
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        line=dict(color='#1DB954'),
        fillcolor='rgba(29, 185, 84, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        title="Profil Audio",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)