# Capstone-Netflix-Recommendation
A multi-approach machine learning recommendation engine that analyzes Netflix movie ratings  and generates personalized movie recommendations using content-based filtering, collaborative  filtering, and SVD-based model techniques. Identifies popular genres and predicts user preferences  to suggest the best-suited movies per genre.

# Netflix Recommendation Engine

A comprehensive machine learning recommendation system that analyzes movie ratings and user preferences to generate personalized movie recommendations using multiple collaborative and content-based filtering techniques.

## 📋 Project Overview

This capstone project implements a **multi-approach recommendation engine** inspired by real-world systems used by Netflix and Prime Video. It combines popularity analysis, content-based similarity, and collaborative filtering to provide data-driven movie recommendations.

### 🎯 Core Objectives

1. **Identify Popular & Liked Genres** – Analyze aggregated ratings to determine which genres are most enjoyed
2. **Recommend Best-Suited Movies Per Genre** – Generate personalized, user-specific movie suggestions for each genre
3. **Analyze Genre Quality** – Determine which genres have received the best and worst ratings
4. **Implement Advanced Filtering** – Apply content-based similarity, collaborative filtering, and model-based approaches

## ✨ Features

### Data Analysis & Exploration
- Load and validate 27,278 movies and 1,048,575 ratings
- Explore genre distribution and rating patterns
- Generate statistical summaries of movie popularity and user engagement

### Genre-Based Recommendations
- Identify the most popular and highest-rated genres
- Calculate mean ratings across all genres
- Visualize genre performance trends

### Personalized Recommendations (Rule-Based)
- Generate weighted genre scores tailored to individual users
- Recommend the best movie for each user in every genre
- Filter recommendations by minimum rating count for quality assurance

### Content-Based Filtering
- **Multi-Hot Encoding** – Convert genre information into binary vectors
- **Cosine Similarity** – Compute movie-to-movie similarity based on shared genres
- **Content Recommendations** – Find similar movies based on genre overlap

### Collaborative Filtering
- **User-Item Correlation** – Measure relationships between users based on rating patterns
- **Item-Based Filtering** – Recommend movies based on user similarity
- **Correlation-Based Scoring** – Predict user preferences using historical rating data

### Model-Based Collaborative Filtering (SVD)
- **Singular Value Decomposition (SVD)** – Uncover latent factors in user-movie interactions
- **Train-Test Validation** – 80:20 split with RMSE and MAE evaluation metrics
- **Predictive Ratings** – Forecast user ratings for unseen movies
- **Personalized SVD Recommendations** – Top-N movie suggestions based on predicted scores

## 🛠 Tech Stack

**Language:** Python 3.7+

**Core Libraries:**
- `pandas` – Data manipulation and analysis
- `numpy` – Numerical computing
- `scikit-learn` – Machine learning utilities (MultiLabelBinarizer, cosine_similarity)
- `scikit-surprise` – Collaborative filtering and SVD implementation
- `matplotlib` – Data visualization
- `seaborn` – Statistical plotting
- `IPython` – Jupyter notebook support

## 📦 Installation

### Prerequisites
- Python 3.7 or higher
- Jupyter Notebook or Google Colab

### Setup Instructions

1. **Clone the repository:**
```bash
   git clone https://github.com/yourusername/netflix-recommendation-engine.git
   cd netflix-recommendation-engine
```

2. **Install dependencies:**
```bash
   pip install pandas numpy matplotlib seaborn scikit-learn scikit-surprise
```

3. **Prepare data files:**
   - Ensure `movies.csv` and `ratings.csv` are in the project directory
   - Expected format:
     - `movies.csv`: movieId, title, genres
     - `ratings.csv`: userId, movieId, rating, timestamp

## 🚀 Usage

### Running the Notebook

1. **Open the Jupyter Notebook:**
```bash
   jupyter notebook Capstone_Project__Netflix_.ipynb
```

2. **Execute cells sequentially:**
   - **Step 1:** Import required libraries
   - **Step 2:** Load movies and ratings data
   - **Step 3:** Analyze genre trends and popularity
   - **Step 4:** Generate rule-based recommendations
   - **Step 5:** Apply content-based filtering (multi-hot + cosine similarity)
   - **Step 6:** Implement collaborative filtering (user-item correlation)
   - **Step 7:** Train SVD model for predictive recommendations
   - **Step 8:** Generate personalized recommendations across all methods

### Key Functions

#### `genre_recommendations(user_id)`
Generates the best-suited movie for a user in each genre using weighted scoring.

```python
# Example usage
recommendations = genre_recommendations(user_id=1)
print(recommendations)
```

#### `content_based_recommend(movie_id, top_n=10)`
Finds similar movies based on shared genres using cosine similarity.

```python
similar_movies = content_based_recommend(movie_id=1, top_n=10)
print(similar_movies)
```

#### `svd_recommend(user_id, top_n=10)`
Predicts ratings and recommends movies using SVD-based collaborative filtering.

```python
svd_recommendations = svd_recommend(user_id=1, top_n=10)
print(svd_recommendations)
```

## 📊 Example Output

### Genre Popularity Analysis
