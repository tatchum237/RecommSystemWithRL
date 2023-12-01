import pandas as pd
import numpy as np
import spacy
from sklearn.decomposition import PCA

from dotenv import load_dotenv, find_dotenv


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from googleapiclient.discovery import build
import aiotube

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from fake_useragent import UserAgent
from urllib.parse import unquote
import time

import h5py
 



# Load pre-trained word embeddings from spaCy
NLP = spacy.load("en_core_web_sm")


def load_data(file_path):
    # Load CSV file
    data = pd.read_csv(file_path)

    # Preprocess data (basic)
    data = data.rename(columns = {'Counrty':'Country'})

    return data


def embed_genre_columns(genres):
    # Function to embed the tokenized name of the relavant genre column
    def embed_genre(genre):
        return NLP(genre).vector  # Embed the tokenized genre name

    # Apply the embedding function to each row
    genre_embeddings = pd.DataFrame(genres.apply(lambda x: embed_genre(x[0]), axis=1))

    # Split embeddings into proper columns
    genre_embeddings[[f"genre_embedding_{i}" for i in range(genre_embeddings.iloc[0,0].shape[0])]] = genre_embeddings[0].values.tolist()
    del genre_embeddings[0]
    
    return genre_embeddings

def preprocess_fast(data):
    
    # preprocessing without embeddings or title
    features = data.iloc[:, 1:] 
    return features
    

def preprocess_without_title(data):
    features_file = "data/features_without_title.h5"

    # Check if the preprocessed features file already exists
    if os.path.exists(features_file):
        print("Loading preprocessed features without title from file.")
        features = pd.read_hdf(features_file)
        return features
    else:
        print("Preprocessing features without title.")

        # Separate columns into decades and genres
        decade_columns = data.columns[1:6]
        genre_columns = data.columns[6:]

        # Apply PCA for dimensionality reduction on decades
        pca_decades = PCA(n_components=2)
        reduced_decades = pd.DataFrame(pca_decades.fit_transform(data[decade_columns]), columns=['PCA_Decade1', 'PCA_Decade2'])

        # Derive vector embeddings of one-hot encoded genres
        genres = pd.DataFrame(data[genre_columns].eq(1, axis=0).idxmax(axis=1))
        genre_embeddings = embed_genre_columns(genres)

        # Concatenate reduced decades and genre embeddings
        features = pd.concat([reduced_decades, genre_embeddings], axis=1)

        # Save the preprocessed features to an HDF5 file
        features.to_hdf(features_file, key='data', mode='w', complevel=1, complib='zlib', format='table')
        
        return features



def preprocess_with_title(data):
    features_file = "data/features_with_title.h5"

    # Check if 'features_with_title.h5' already exists
    if os.path.exists(features_file):
        print("Loading features_with_title.h5...")
        features = pd.read_hdf(features_file, key='data')
        return features

    # Separate columns into decades and genres
    decade_columns = data.columns[1:6]
    genre_columns = data.columns[6:]

    # Apply PCA for dimensionality reduction on decades
    pca_decades = PCA(n_components=2)
    reduced_decades = pd.DataFrame(pca_decades.fit_transform(data[decade_columns]), columns=['PCA_Decade1', 'PCA_Decade2'])

    # Apply PCA for dimensionality reduction on genres
    pca_genres = PCA(n_components=2)
    reduced_genres = pd.DataFrame(pca_genres.fit_transform(data[genre_columns]), columns=['PCA_Genre1', 'PCA_Genre2'])

    # Tokenize and pad song titles
    max_title_length = data['Title'].apply(len).max()  # Find the maximum title length
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data['Title'])
    title_sequences = tokenizer.texts_to_sequences(data['Title'])
    padded_titles = pad_sequences(title_sequences, maxlen=max_title_length, padding='post')

    # Average word embeddings for each title
    title_embeddings = []
    for title_sequence in padded_titles:
        title_tokens = [tokenizer.index_word.get(token, '') for token in title_sequence]
        title_vectors = [NLP(word).vector for word in title_tokens if word]
        if title_vectors:
            title_vector = np.mean(title_vectors, axis=0)
        else:
            title_vector = np.zeros_like(title_vectors[0]) if title_vectors else np.zeros(NLP.vocab.vectors.shape[1])
        title_embeddings.append(title_vector)

    # Create DataFrame for title embeddings
    embedding_dim = title_embeddings[0].shape[0]
    title_columns = [f'Title_Embedding_{i+1}' for i in range(embedding_dim)]
    title_embeddings_df = pd.DataFrame(title_embeddings, columns=title_columns)

    # Concatenate the reduced features, title embeddings, and original DataFrame
    features = pd.concat([reduced_decades, reduced_genres, title_embeddings_df], axis=1)

    # Save the DataFrame to an HDF5 file
    features.to_hdf(features_file, key='data', mode='w', complevel=1, complib='zlib', format='table')
    print(f"Features saved to {features_file}")

    return features


# Function to simulate fetching the next song context in the online setting
def get_next_song_context(songs, features, chosen_action=0):
    try:
        
        # Get the title and context based on the chosen action
        title = songs.iloc[chosen_action, 0]
        context = features.iloc[chosen_action].values

        return title, context
    except IndexError:
        return None, None
    

def search_youtube(api_key, query):
    youtube = build("youtube", "v3", developerKey=api_key)
    request = youtube.search().list(q=query, part="snippet", type="video", maxResults=1)
    response = request.execute()
    items = response.get("items", [])
    
    if items:
        return f"https://www.youtube.com/watch?v={items[0]['id']['videoId']}"
    else:
        print(f"No YouTube link found for query: {query}")
        return None
    
def search_youtube_alt(query):
    search = aiotube.Search.video(query)

    if search:
        return search.metadata['url']
    else:
        print(f"No YouTube link found for query: {query}")
        return None

def search_duck(query):
    headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:84.0) Gecko/20100101 Firefox/84.0",
    }

    page = requests.get(f'https://duckduckgo.com/html/?q={query}', headers=headers).text
    soup = BeautifulSoup(page, 'html.parser').find("a", class_="result__url", href=True)

    return soup['href']


def search_duck_selenium(query):
    ua = UserAgent()
    user_agent = ua.random
    
    # Create Chromeoptions instance 
    options = webdriver.ChromeOptions() 
    
    # Set up the webdriver with a fake user agent
    options.add_argument(f"user-agent={user_agent}")
    
    # Adding argument to disable the AutomationControlled flag 
    options.add_argument("--disable-blink-features=AutomationControlled") 
    
    # Exclude the collection of enable-automation switches 
    options.add_experimental_option("excludeSwitches", ["enable-automation"]) 
    
    # Turn-off userAutomationExtension 
    options.add_experimental_option("useAutomationExtension", False) 
    
    # Setting the driver path
    driver = webdriver.Chrome(options=options) 
    
    # Changing the property of the navigator value for webdriver to undefined 
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})") 
    

    try:
        # Navigate to the search URL
        search_url = f'https://duckduckgo.com/html/?q={query}'
        driver.get(search_url)

        # Wait 1s on the webpage before trying anything 
        time.sleep(1) 

        # Wait for the results to load
        WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CLASS_NAME, 'result__url')))

        # Wait 2s before scrolling down 100px 
        time.sleep(2) 
        driver.execute_script('window.scrollTo(0, 100)') 
        
        # Extract the link from the first result
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        result_link = soup.find("a", class_="result__url", href=True)

        if result_link:
            # Extract the href attribute (redirected URL)
            redirected_url = result_link['href']
            
            # Decode the URL (remove URL encoding)
            actual_url = unquote(redirected_url.split('uddg=')[1])

            # Extract the essential part of the YouTube URL
            essential_url = actual_url.split('&')[0]

            return essential_url
        else:
            print(f"No link found for {query}")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # Close the webdriver
        driver.quit() 
    

def main():
    
    # # Load API key from environment variable
    # load_dotenv(find_dotenv())    
    # api_key = os.getenv("YOUTUBE_API_KEY")
    # if api_key is None:
    #     print("YouTube API key not found. Please set the YOUTUBE_API_KEY environment variable.")
    #     return
    
    data = load_data("data/songs.csv")
    # Check if 'songs_links.csv' already exists
    if os.path.exists("data/songs_links.csv"):
        print("The file 'songs_links.csv' already exists. Checking for completeness.")
        songs_links = pd.read_csv("data/songs_links.csv")
        
        
        # Check if 'songs_links.csv' is complete
        if len(songs_links) >= len(data):
            print("The file 'songs_links.csv' is complete. Skipping YouTube search.")
            return
        else:
            last_index = songs_links.index[-1] + 1  # Get the last index and add 1 to start from the next row
            print(f"Resuming search from index {last_index}")
    else:
        # Initialize an empty DataFrame to store links
        songs_links = pd.DataFrame(columns=['Title', 'link'])

    # Iterate over rows and search for YouTube links
    for index, row in data.iloc[last_index:].iterrows():
        title = row['Title']
        year_column = pd.DataFrame(row[['(1980s)', '(1990s)', '(2000s)', '(2010s)', '(2020s)']]).transpose().eq(1, axis=0).idxmax(axis=1)[index]
        query = f"song called: {title} {year_column} inurl:youtube -lyrics" 
        
        try:
            link = search_duck_selenium(query)
        except Exception as e:
            print(f"Error searching YouTube for '{title}': {str(e)}")
            break

        songs_links = pd.concat([songs_links, pd.DataFrame({'Title': [title], 'link': [link]})], ignore_index=True)

    # Save the DataFrame to a CSV file
    songs_links.to_csv("data/songs_links.csv", index=False)
    print("YouTube search and link extraction completed.")
    
    
    
if __name__ == "__main__":
    main()
