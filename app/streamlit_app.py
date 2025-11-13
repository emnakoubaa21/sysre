"""
Application Streamlit - Système de Recommandation Musicale - Version Professionnelle
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# Imports locaux
from config import *
from models import RecommendationSystem
from utils import *


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def get_genre_emoji(genre: str) -> str:
    """Retourne un emoji correspondant au genre musical"""
    return GENRE_EMOJIS.get(genre, '🎵')


def highlight_liked_songs(history):
    """Crée un DataFrame formaté de l'historique avec mise en évidence des chansons aimées"""
    df_data = []
    for song in history:
        df_data.append({
            'Titre': song.get('title', 'Unknown'),
            'Artiste': song.get('artist', 'Unknown'),
            'Genre': song.get('genre', 'Unknown'),
            'Album': song.get('album', 'N/A'),
            'Statut': '❤️ Aimé' if song.get('liked', 0) == 1 else '👍 Écouté'
        })
    return pd.DataFrame(df_data)


def plot_model_comparison(stats):
    """Affiche un graphique de comparaison des performances des modèles"""
    models = ['Content-Based', 'Collaborative', 'Hybrid']
    metrics_data = {
        'Modèle': [],
        'Precision@10 (%)': [],
        'Coverage (%)': [],
        'Diversité (%)': []
    }
    
    for model_key, model_name in [('content_based', 'Content-Based'), 
                                    ('collaborative', 'Collaborative'), 
                                    ('hybrid', 'Hybrid')]:
        if model_key in stats:
            model_stats = stats[model_key]
            metrics_data['Modèle'].append(model_name)
            metrics_data['Precision@10 (%)'].append(model_stats.get('precision_at_10', 0) * 100)
            metrics_data['Coverage (%)'].append(model_stats.get('coverage', 0) * 100)
            metrics_data['Diversité (%)'].append(model_stats.get('diversity_genre', 0) * 100)
    
    if metrics_data['Modèle']:
        df_metrics = pd.DataFrame(metrics_data)
        
        fig = px.bar(
            df_metrics,
            x='Modèle',
            y=['Precision@10 (%)', 'Coverage (%)', 'Diversité (%)'],
            title="Comparaison des Performances",
            barmode='group',
            color_discrete_map={
                'Precision@10 (%)': '#FF6B6B',
                'Coverage (%)': '#4ECDC4',
                'Diversité (%)': '#45B7D1'
            }
        )
        
        fig.update_layout(
            xaxis_title="Modèle",
            yaxis_title="Score (%)",
            legend_title="Métrique",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# CONFIGURATION DE LA PAGE
# =============================================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=LAYOUT,
    initial_sidebar_state=SIDEBAR_STATE
)

# CSS personnalisé
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# INITIALISATION
# =============================================================================

@st.cache_resource(ttl=3600)
def load_system():
    """Charge le système de recommandation"""
    system = RecommendationSystem()
    success = system.load_all()
    return system, success


# Chargement du système (SANS afficher les messages de debug)
rec_system, load_success = load_system()

if not load_success:
    st.error("❌ Impossible de charger les modèles. Vérifiez que tous les fichiers sont présents.")
    st.stop()


# =============================================================================
# SIDEBAR - SÉLECTION UTILISATEUR ET PARAMÈTRES
# =============================================================================

st.sidebar.title("⚙️ Configuration")

# Sélection de l'utilisateur
available_users = sorted(rec_system.train_df['user_id'].unique())
selected_user = st.sidebar.selectbox(
    "👤 Sélectionner un utilisateur",
    available_users,
    index=0
)

st.sidebar.markdown("---")

# Nombre de recommandations
n_recommendations = st.sidebar.slider(
    "🔢 Nombre de recommandations",
    min_value=MIN_RECOMMENDATIONS,
    max_value=MAX_RECOMMENDATIONS,
    value=DEFAULT_N_RECOMMENDATIONS,
    step=1
)

st.sidebar.markdown("---")

# Paramètres du modèle Hybrid
st.sidebar.subheader("🔀 Modèle Hybrid")
st.sidebar.caption("Ajuster les poids des modèles")

alpha = st.sidebar.slider(
    "Content-Based (α)",
    min_value=0.0,
    max_value=1.0,
    value=DEFAULT_ALPHA,
    step=0.1
)

beta = st.sidebar.slider(
    "Collaborative (β)",
    min_value=0.0,
    max_value=1.0,
    value=DEFAULT_BETA,
    step=0.1
)

# Vérification que α + β = 1
if abs(alpha + beta - 1.0) > 0.01:
    st.sidebar.warning("⚠️ La somme α + β doit égaler 1.0")
    total = alpha + beta
    if total > 0:
        alpha = alpha / total
        beta = beta / total

st.sidebar.markdown("---")

# Bouton de rechargement
if st.sidebar.button("🔄 Recharger les données", use_container_width=True):
    st.cache_resource.clear()
    st.rerun()

st.sidebar.markdown("---")

# Affichage des paramètres actuels
st.sidebar.info(f"""
**Paramètres actuels:**
- Utilisateur: {selected_user}
- Recommandations: {n_recommendations}
- α (Content): {alpha:.1f}
- β (Collaborative): {beta:.1f}
""")


# =============================================================================
# HEADER
# =============================================================================

st.markdown(f"<div class='main-header'>{APP_TITLE}</div>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; color: #666; margin-bottom: 2rem;'>
    Système intelligent de recommandation musicale basé sur trois approches complémentaires
</div>
""", unsafe_allow_html=True)


# =============================================================================
# ONGLETS PRINCIPAUX
# =============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Accueil",
    "🎵 Recommandations",
    "📊 Comparaison",
    "📈 Statistiques"
])


# =============================================================================
# ONGLET 1 : ACCUEIL
# =============================================================================

with tab1:
    st.header(f"👤 Profil de l'utilisateur {selected_user}")
    
    # Historique d'écoute
    st.subheader("🎧 Historique récent")
    
    history = rec_system.get_user_history(selected_user, n=20)
    
    if not history:
        st.warning(MSG_NO_HISTORY)
    else:
        # Statistiques de l'utilisateur
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total écoutes", len(history))
        
        with col2:
            liked_count = sum(1 for h in history if h.get('liked', 0) == 1)
            st.metric("Chansons aimées", liked_count)
        
        with col3:
            genres = [h['genre'] for h in history if h.get('genre') and h['genre'] != 'Unknown']
            if genres:
                user_genres = pd.Series(genres).value_counts()
                top_genre = user_genres.index[0] if len(user_genres) > 0 else "N/A"
                genre_emoji = get_genre_emoji(top_genre)
                st.metric("Genre préféré", f"{genre_emoji} {top_genre}")
            else:
                st.metric("Genre préféré", "N/A")
        
        with col4:
            artists = [h['artist'] for h in history if h.get('artist') and h['artist'] != 'Unknown']
            unique_artists = len(set(artists))
            st.metric("Artistes écoutés", unique_artists)
        
        st.markdown("---")
        
        # Affichage de l'historique
        st.subheader("📜 Dernières écoutes")
        
        # Tableau avec mise en forme
        df_history = highlight_liked_songs(history[:10])
        st.dataframe(df_history, use_container_width=True, hide_index=True)
        
        # Graphique de distribution des genres
        if genres:
            st.subheader("📊 Distribution des genres écoutés")
            genre_counts = pd.Series(genres).value_counts()
            
            fig = px.pie(
                values=genre_counts.values,
                names=genre_counts.index,
                title="Répartition des genres dans l'historique",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# ONGLET 2 : RECOMMANDATIONS
# =============================================================================

with tab2:
    st.header("🎵 Générer des recommandations")
    
    # Sélection du modèle
    model_choice = st.radio(
        "Choisir un modèle",
        ["Content-Based", "Collaborative", "Hybrid"],
        horizontal=True
    )
    
    # Bouton de génération
    if st.button("🎵 Générer les recommandations", type="primary", use_container_width=True):
        
        with st.spinner(MSG_GENERATING_RECS):
            
            if model_choice == "Content-Based":
                recs = rec_system.recommend_content_based(selected_user, n=n_recommendations)
                recommendations = [
                    {**rec_system.get_song_info(song_id), 'score': score}
                    for song_id, score in recs
                ]
                
            elif model_choice == "Collaborative":
                recs = rec_system.recommend_collaborative(selected_user, n=n_recommendations)
                recommendations = [
                    {**rec_system.get_song_info(song_id), 'score': score}
                    for song_id, score in recs
                ]
                
            else:  # Hybrid
                recs = rec_system.recommend_hybrid(selected_user, n=n_recommendations, alpha=alpha, beta=beta)
                recommendations = [
                    {**rec_system.get_song_info(song_id), 'score': score}
                    for song_id, score in recs
                ]
        
        # Affichage des résultats
        if not recommendations:
            st.error(f"❌ Aucune recommandation disponible avec le modèle {model_choice}")
        else:
            st.success(f"✅ {len(recommendations)} recommandations générées avec {model_choice}")
            
            # Mode d'affichage
            display_mode = st.radio(
                "Mode d'affichage",
                ["Cartes", "Tableau"],
                horizontal=True,
                key="display_mode"
            )
            
            # ✅ CORRECTION : Afficher selon le mode choisi
            if display_mode == "Cartes":
                # Affichage en cartes
                for i, song in enumerate(recommendations, 1):
                    st.markdown(f"### 🎵 Recommandation #{i}")
                    display_song_card(song, show_score=True, show_details=True)
            
            else:  # Tableau
                # ✅ CORRECTION : Afficher le tableau correctement
                st.markdown(f"### 📊 Recommandations {model_choice}")
                display_recommendations_table(recommendations)


# =============================================================================
# ONGLET 3 : COMPARAISON
# =============================================================================

with tab3:
    st.header("📊 Comparaison des modèles")
    
    st.markdown("""
    Comparez les recommandations des trois modèles côte à côte pour le même utilisateur.
    """)
    
    if st.button("🔄 Comparer les 3 modèles", type="primary", use_container_width=True):
        
        with st.spinner("Génération des recommandations pour les 3 modèles..."):
            all_recs = rec_system.get_recommendations_all_models(selected_user, n=n_recommendations)
        
        # Affichage en colonnes
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🎸 Content-Based")
            if all_recs['content']:
                for i, song in enumerate(all_recs['content'][:5], 1):
                    st.markdown(f"**{i}. {song['title']}**")
                    st.caption(f"{song['artist']} - {song['genre']}")
                    score = song['score']
                    if score > 1.0:
                        score = min(score / 10.0, 1.0)
                    st.caption(f"Score: {score:.3f}")
                    st.markdown("---")
            else:
                st.warning("Aucune recommandation")
        
        with col2:
            st.subheader("🤝 Collaborative")
            if all_recs['collaborative']:
                for i, song in enumerate(all_recs['collaborative'][:5], 1):
                    st.markdown(f"**{i}. {song['title']}**")
                    st.caption(f"{song['artist']} - {song['genre']}")
                    score = song['score']
                    if score > 1.0:
                        score = min(score / 10.0, 1.0)
                    st.caption(f"Score: {score:.3f}")
                    st.markdown("---")
            else:
                st.warning("Aucune recommandation (cold start)")
        
        with col3:
            st.subheader("🔀 Hybrid")
            if all_recs['hybrid']:
                for i, song in enumerate(all_recs['hybrid'][:5], 1):
                    st.markdown(f"**{i}. {song['title']}**")
                    st.caption(f"{song['artist']} - {song['genre']}")
                    score = song['score']
                    if score > 1.0:
                        score = min(score / 10.0, 1.0)
                    st.caption(f"Score: {score:.3f}")
                    st.markdown("---")
            else:
                st.warning("Aucune recommandation")
        
        st.markdown("---")
        
        # ✅ AJOUT : Graphique de distribution des genres
        st.subheader("📊 Distribution des genres par modèle")
        plot_genre_distribution(all_recs)
        
        # ✅ AMÉLIORATION : Analyse de diversité avec explication
        st.subheader("🎭 Analyse de diversité")
        
        # ✅ AJOUT : Explication claire
        st.info("""
        **📌 La diversité mesure la variété des genres musicaux dans les recommandations.**
        
        - **Score élevé (>70%)** = Beaucoup de genres différents → Découverte
        - **Score moyen (40-70%)** = Équilibre entre variété et cohérence
        - **Score faible (<40%)** = Genres similaires → Précision ciblée
        
        *Formule : Diversité = (Nombre de genres uniques / Nombre total de chansons) × 100%*
        """)
        
        col1, col2, col3 = st.columns(3)
        
        for col, (model_name, recs) in zip([col1, col2, col3], all_recs.items()):
            with col:
                if recs:
                    genres = [r['genre'] for r in recs if r.get('genre') != 'Unknown']
                    if genres:
                        unique_genres = len(set(genres))
                        diversity = unique_genres / len(genres)
                        
                        st.metric(
                            f"{model_name.title()}",
                            f"{diversity:.1%}",
                            help=f"{unique_genres} genres différents sur {len(genres)} chansons"
                        )
                    else:
                        st.metric(f"{model_name.title()}", "N/A")
                else:
                    st.metric(f"{model_name.title()}", "N/A")


# =============================================================================
# ONGLET 4 : STATISTIQUES
# =============================================================================

with tab4:
    st.header("📈 Statistiques des modèles")
    
    # Charger les statistiques
    stats = rec_system.get_model_stats()
    
    if not stats:
        st.warning("Aucune statistique disponible")
    else:
        # Résumé général
        st.subheader("📋 Résumé du projet")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Utilisateurs",
                stats.get('dataset', {}).get('n_users', 'N/A')
            )
        
        with col2:
            st.metric(
                "Chansons",
                stats.get('dataset', {}).get('n_songs', 'N/A')
            )
        
        with col3:
            st.metric(
                "Interactions (train)",
                f"{stats.get('dataset', {}).get('n_interactions_train', 0):,}"
            )
        
        st.markdown("---")
        
        # Performances des modèles
        st.subheader("🎯 Performances")
        
        # Graphique de comparaison
        plot_model_comparison(stats)
        
        st.markdown("---")
        
        # Détails par modèle
        st.subheader("📊 Détails par modèle")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🎸 Content-Based")
            if 'content_based' in stats:
                cb_stats = stats['content_based']
                st.metric("Precision@10", f"{cb_stats.get('precision_at_10', 0):.1%}")
                st.metric("Coverage", f"{cb_stats.get('coverage', 0):.1%}")
                st.metric("Diversité", f"{cb_stats.get('diversity_genre', 0):.1%}")
                st.metric("Taux de succès", f"{cb_stats.get('success_rate', 0):.1%}")
        
        with col2:
            st.markdown("### 🤝 Collaborative")
            if 'collaborative' in stats:
                collab_stats = stats['collaborative']
                st.metric("Precision@10", f"{collab_stats.get('precision_at_10', 0):.1%}")
                st.metric("Precision (Novelties)", f"{collab_stats.get('precision_at_10_novelties', 0):.1%}")
                st.metric("Coverage", f"{collab_stats.get('coverage', 0):.1%}")
                st.metric("Taux de succès", f"{collab_stats.get('success_rate', 0):.1%}")
        
        with col3:
            st.markdown("### 🔀 Hybrid")
            if 'hybrid' in stats:
                hybrid_stats = stats['hybrid']
                st.metric("Precision@10", f"{hybrid_stats.get('precision_at_10', 0):.1%}")
                st.metric("Precision (Novelties)", f"{hybrid_stats.get('precision_at_10_novelties', 0):.1%}")
                st.metric("Coverage", f"{hybrid_stats.get('coverage', 0):.1%}")
                st.metric("Taux de succès", f"{hybrid_stats.get('success_rate', 0):.1%}")
        
        st.markdown("---")
        
        # Recommandation
        if 'recommendation' in stats:
            st.subheader("🏆 Recommandation")
            
            st.success(f"""
            **Modèle recommandé pour production:** {stats['recommendation'].get('production_model', 'Hybrid')}
            
            **Raisons:**
            """)
            
            reasons = stats['recommendation'].get('reasons', [])
            for reason in reasons:
                st.markdown(f"- ✅ {reason}")


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 2rem 0;'>
    🎵 Système de Recommandation Musicale | Développé avec Streamlit<br>
    📊 Dataset: MCRec-30M | 🤖 Modèles: Content-Based, Collaborative, Hybrid
</div>
""", unsafe_allow_html=True)