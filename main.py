import pandas as pd
import numpy as np
import string
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import re

# Download necessary NLTK data
nltk.download('punkt')


def load_data(file_path):
    """
    Load data from a CSV file.

    Args:
    file_path (str): Path to the CSV file.

    Returns:
    DataFrame: Loaded data as a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        return None
    except pd.errors.ParserError:
        print("Error: The file could not be parsed.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def clean_data(df):
    """
    Clean the data by removing null values and unnecessary columns.

    Args:
    df (DataFrame): Input DataFrame.

    Returns:
    DataFrame: Cleaned DataFrame.
    """
    try:
        # Remove null values
        df = df.dropna()
        # Keep only the 'review' column
        if 'review' not in df.columns:
            raise ValueError("The required 'review' column is missing in the input data.")
        df = df[['review']]
        return df
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during data cleaning: {e}")
        return None


def preprocess_text(text):
    """
    Preprocess the text by converting to lowercase, removing punctuation, numbers, and extra whitespace.

    Args:
    text (str): Input text.

    Returns:
    str: Preprocessed text.
    """
    try:
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    except TypeError as te:
        print(f"TypeError: The input text is not a string: {te}")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred during text preprocessing: {e}")
        return ""


def analyze_sentiment_vader(text):
    """
    Analyze the sentiment of the text using VADER.

    Args:
    text (str): Input text.

    Returns:
    str: Sentiment label ('positive', 'negative', 'neutral').
    """
    try:
        analyzer = SentimentIntensityAnalyzer()
        sentiment_score = analyzer.polarity_scores(text)
        if sentiment_score['compound'] >= 0.05:
            return 'positive'
        elif sentiment_score['compound'] <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    except Exception as e:
        print(f"An error occurred during sentiment analysis: {e}")
        return 'neutral'


def save_to_csv(df, file_path):
    """
    Save the DataFrame to a CSV file.

    Args:
    df (DataFrame): Input DataFrame.
    file_path (str): Path to save the CSV file.
    """
    try:
        df.to_csv(file_path, index=False)
        print(f"File saved successfully at {file_path}.")
    except Exception as e:
        print(f"Error saving file: {e}")


def main(input_file, output_file, summary_file):
    """
    Main function to load, clean, preprocess, analyze, and save data.

    Args:
    input_file (str): Path to the input CSV file.
    output_file (str): Path to the output CSV file.
    summary_file (str): Path to the summary CSV file.
    """
    # Load data
    df = load_data(input_file)
    if df is None:
        return

    # Clean data
    df = clean_data(df)
    if df is None:
        return

    # Preprocess text
    try:
        df['review'] = df['review'].apply(preprocess_text)
    except Exception as e:
        print(f"An error occurred during text preprocessing: {e}")
        return

    # Analyze sentiment
    try:
        df['sentiment'] = df['review'].apply(analyze_sentiment_vader)
    except Exception as e:
        print(f"An error occurred during sentiment analysis: {e}")
        return

    # Create summary report
    try:
        sentiment_summary = df['sentiment'].value_counts().reset_index()
        sentiment_summary.columns = ['sentiment', 'count']
    except Exception as e:
        print(f"An error occurred while creating the summary report: {e}")
        return

    # Save processed data and summary report to CSV files
    save_to_csv(df, output_file)
    save_to_csv(sentiment_summary, summary_file)


if __name__ == "__main__":
    input_file = 'user_review.csv'
    output_file = 'processed_reviews.csv'
    summary_file = 'sentiment_summary.csv'
    main(input_file, output_file, summary_file)
