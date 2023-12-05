# AI Music Recommender System

## Introduction

This project implements an interactive music recommender system using a contextual Multi-Armed Bandit, in particular the LinUCB algorithm for online learning. The recommender system suggests songs to users and updates its model based on user feedback. The project also includes a web scraping component to fill in missing YouTube links for songs.

## Files

The project consists of the following main files:

- **model.py**: Defines the LinUCB class, which implements the LinUCB algorithm for online learning.


- **utils.py**: Contains utility functions for data preprocessing, embedding, and YouTube search.


- **interactive_scraping.ipynb**: Jupyter notebook for interactively scraping missing YouTube links.


- **train.py**: Script for training the LinUCB model on user interactions.


- **train.ipynb**: Jupyter notebook for interactive training of the LinUCB model with a graphical user interface.

See [project structure](#project-structure) below to have a more complete view of the file structure.

## How to Run

To run the project, follow these steps (steps 2. and 3 should be unecessary if the file `songs_links.csv` is available and complete):

1. Install the required dependencies by running:

```bash
pip install -r requirements.txt
```
2. If needed, run `utils.py`, which contains the main scraping script to look for YouTube links of the songs.

3. Execute the interactive scraping notebook `interactive_scraping.ipynb` to complete missing YouTube link (if any).

4. Run the training script `train.py` to train the LinUCB model on user interactions.

```bash
python train.py
```

5. Alternatively, you can use the interactive training notebook `train.ipynb` for a graphical user interface.

## Project Structure

```plaintext
|   .gitignore
|   interactive_scraping.ipynb
|   LICENSE
|   model.py
|   README.md
|   requirements.txt
|   train.py
|   train.ipynb
|   utils.py
|
+---.ipynb_checkpoints
|       train-checkpoint.ipynb
|
\---data
        features_without_title.h5
        features_with_title.h5
        songs.csv
        songs_links.csv
```

## Song Title and Genre Embeddings

In this project, we provide alternative preprocessing functions within `utils.py` that enable the generation of dense representations for song titles and genres. These embeddings serve the purpose of enhancing the training signal provided to the agent, ultimately contributing to a more robust and effective learning process.

### Preprocessing Functions

1. **Fast Preprocessing (`preprocess_fast`):**
   This function focuses on a basic preprocessing approach without incorporating embeddings for song titles. It efficiently extracts features from the available data as-is, only omitting the titles.

2. **Preprocessing without Title (`preprocess_without_title`):**
   This function extends the preprocessing by still excluding song titles but including genre embeddings. It leverages PCA for dimensionality reduction on decades, providing a simplified feature set.

3. **Preprocessing with Title (`preprocess_with_title`):**
   In contrast, this function incorporates embeddings for song titles using techniques such as tokenization, padding, and averaging word embeddings but ignores genre embeddings. It also includes PCA for both decades and genres, providing a comprehensive feature set that captures title-related information.

Except for the first preprocessing function, the feature vectors generated are cached in the HDF5 files `features_with_title.h5` and  `features_without_title.h5`, in order to trade CPU usage for I/O during the preprocessing phase.

### Leveraging Dense Representations

The inclusion of song title and genre embeddings allows the model to capture more nuanced relationships and patterns in the data. By representing titles and genres in a dense format, the agent receives a stronger and more informative(or at least, more geometrically sound) training signal. This richer input helps the agent make better-informed decisions during the exploration-exploitation process. Ideally, we would also provide vector embeddings of the actual song or music video content but it would be beyond the scope of this simple demo, not to mention computationally redhibitory.

### Choosing the Right Preprocessing Strategy

The choice of preprocessing strategy depends on the specific (mostly performance-related) requirements and goals of the application. If a more lightweight approach is desired, the function excluding title and genre embeddings can be employed. On the other hand, if a more detailed and nuanced representation is needed, the functions incorporating title/genre embeddings and additional PCA can be selected at the cost of more compute usage during inference/training. We made sure that those options are all available to the end-user. Feel free to experiment with each and all of them.

## Acknowledgments

- This project uses the [LinUCB algorithm](https://proceedings.mlr.press/v15/chu11a/chu11a.pdf) for [online learning](https://www.wikiwand.com/en/Online_machine_learning).
- Web scraping for missing links is implemented using [Selenium](https://www.selenium.dev/) and [BeautifulSoup](https://beautiful-soup-4.readthedocs.io/en/latest/) and the results, saved in the `songs_links.csv` file.
- Initial music data is sourced from the provided `songs.csv` file.

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.
