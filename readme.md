# Sentiment Analysis Pipeline

This repository contains a Python script for performing sentiment analysis on user reviews using the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis tool. The script loads data from a CSV file, preprocesses the text, analyzes the sentiment, and saves the results to a new CSV file along with a summary of sentiment counts.

## Features

- Load data from a CSV file
- Clean and preprocess text data
- Analyze sentiment using VADER
- Save processed data and sentiment summary to CSV files

## Requirements

- Python 3.6 or higher
- pandas
- numpy
- vaderSentiment
- nltk
- re

## Setup

1. Clone the repository:
   ```sh
   git clone https://github.com/your_username/sentiment-analysis-pipeline.git
   cd sentiment-analysis-pipeline
2. Create and activate a virtual environment (optional but recommended):

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3. Install the required packages:

    ```sh
    pip install pandas numpy vaderSentiment nltk
4. Download necessary NLTK data:

    ```sh
    python -c "import nltk; nltk.download('punkt')"
## Usage
1. Prepare your input CSV file (user_review.csv) with a column named review containing the text data for sentiment analysis.

2. Run the script:

    ```sh
    python sentiment_analysis.py
3. By default, the script expects the input file to be named `user_review.csv`. It will generate two output files: `processed_reviews.csv` and `sentiment_summary.csv`.

## Script Overview

1. `load_data(file_path)`
    Loads data from a CSV file.

    #### Arguments:

    - `file_path` (`str`): Path to the CSV file.
    
    #### Returns:
    
    - `DataFrame`: Loaded data as a pandas DataFrame.

2. `clean_data(df)`
    Cleans the data by removing null values and unnecessary columns.

    #### Arguments:

    - `df (DataFrame)`: Input DataFrame.

    #### Returns:

    - `DataFrame`: Cleaned DataFrame.

3. `preprocess_text(text)`
    Preprocesses the text by converting it to lowercase, removing punctuation, numbers, and extra whitespace.
    
    #### Arguments:

    - `text (str)`: Input text.
    
    #### Returns:

    - `str`: Preprocessed text.

4. `analyze_sentiment_vader(text)`
    Analyzes the sentiment of the text using VADER.

    #### Arguments:

    - `text (str)`: Input text.

    #### Returns:

    - `tr`: Sentiment label (`positive`, `negative`, `neutral`).

5. `save_to_csv(df, file_path)`
    Saves the DataFrame to a CSV file.

    #### Arguments:

    - `df (DataFrame)`: Input DataFrame.
    - `file_path (str)`: Path to save the CSV file.

6. `main(input_file, output_file, summary_file)`
    Main function to load, clean, preprocess, analyze, and save data.

    #### Arguments:

    - `input_file (str)`: Path to the input CSV file.
    - `output_file (str)`: Path to the output CSV file.
    - `summary_file (str)`: Path to the summary CSV file.

## Example
Assuming you have a CSV file named `user_review.csv` with the following content:

```sh
review
"This product is great!"
"I am not happy with this service."
"The experience was okay, nothing special."
```    

Running the script will generate `processed_reviews.csv` and `sentiment_summary.csv` with the analyzed sentiments.
