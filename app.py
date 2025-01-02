import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Your Spotify API credentials
CLIENT_ID = "4231510d9b0643b4b5358315f1addd0c"
CLIENT_SECRET = "da5ad479b0e0425db37c0de6e516529e"
REDIRECT_URI = "http://localhost:8501/callback"  # Replace with your callback URI

# Authenticate using client credentials
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET))

# Load the dataset
df = pd.read_csv('dataset.csv')  # Path to your dataset

# Preprocess data (scale features)
features = ['tempo', 'energy', 'danceability', 'valence', 'loudness']
df = df.dropna(subset=features + ['track_name'])
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)
df_scaled.index = df.index
df = pd.concat([df, df_scaled.add_suffix('_scaled')], axis=1)

# Recommendation function
def recommend_songs(query, criterion='track_name', top_n=5):
    try:
        query = query.lower().strip()  # Case-insensitive and strip spaces
        
        if criterion == 'track_name':
            matches = df[df['track_name'].str.lower() == query]
            if matches.empty:
                return f"No matching track found for '{query}'"
            idx = matches.index[0]
        elif criterion == 'artists':
            matching_artists = df[df['artists'].str.lower().str.contains(query)]
            if matching_artists.empty:
                return f"No matching artist found for '{query}'"
            idx = matching_artists.index[0]
        elif criterion == 'album_name':
            matches = df[df['album_name'].str.lower() == query]
            if matches.empty:
                return f"No matching album found for '{query}'"
            idx = matches.index[0]
        else:
            return "Criterion must be one of ['track_name', 'artists', 'album_name']"

        # Calculate similarities
        similarities = cosine_similarity([df_scaled.iloc[idx]], df_scaled)[0]
        similar_indices = similarities.argsort()[-(top_n+1):][::-1]
        similar_indices = [i for i in similar_indices if i != idx]

        recommendations = df.iloc[similar_indices][['track_name', 'artists', 'album_name', 'popularity']]
        return recommendations

    except Exception as e:
        return f"Error: {str(e)}"

# Spotify track search
def get_spotify_track_url_and_image(track_name):
    result = sp.search(q=track_name, limit=1, type='track')
    if result['tracks']['items']:
        track_info = result['tracks']['items'][0]
        track_url = track_info['external_urls']['spotify']
        image_url = track_info['album']['images'][0]['url']  # Get the first image size (large)
        return track_url, image_url
    else:
        return None, None

# Streamlit UI
st.title("Music Recommendation App")

# User input for song search
query = st.text_input("Enter track name, artist, or album:")
criterion = st.selectbox("Search by", ['track_name', 'artists', 'album_name'])
top_n = st.slider("Number of recommendations", 1, 10, 5)

if query:
    st.write("Searching for recommendations...")
    recommendations = recommend_songs(query, criterion, top_n)
    
    if isinstance(recommendations, pd.DataFrame):
        # Displaying recommendations in a table format
        for _, row in recommendations.iterrows():
            track_url, image_url = get_spotify_track_url_and_image(row['track_name'])

            # Create a row with the song's image and details
            col1, col2 = st.columns([1, 4])  # Two columns: image on the left and details on the right
            with col1:
                if image_url:
                    st.image(image_url, caption=row['track_name'], use_container_width=True)  # Display image without border
            with col2:
                st.markdown(f"**Track:** {row['track_name']}")
                st.markdown(f"**Artist:** {row['artists']}")
                st.markdown(f"**Album:** {row['album_name']}")
                st.markdown(f"**Popularity:** {row['popularity']}")
                if track_url:
                    st.markdown(f"[Play on Spotify]({track_url})")
                else:
                    st.write("Track not found on Spotify.")
    else:
        st.write(recommendations)
