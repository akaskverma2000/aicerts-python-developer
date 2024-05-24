import pandas as pd
import numpy as np
import string
from textblob import TextBlob


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


def clean_data(df):
    """
    Clean the data by removing null values and unnecessary columns.

    Args:
    df (DataFrame): Input DataFrame.

    Returns:
    DataFrame: Cleaned DataFrame.
    """
    # Remove null values
    df = df.dropna()
    # Keep only the 'review' column
    df = df[['review']]
    return df


def preprocess_text(text):
    """
    Preprocess the text by converting to lowercase and removing punctuation.

    Args:
    text (str): Input text.

    Returns:
    str: Preprocessed text.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


def analyze_sentiment(text):
    """
    Analyze the sentiment of the text using TextBlob.

    Args:
    text (str): Input text.

    Returns:
    str: Sentiment label ('positive', 'negative', 'neutral').
    """
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
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
        print(f"Error: {e}")


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
    # Preprocess text
    df['review'] = df['review'].apply(preprocess_text)
    # Analyze sentiment
    df['sentiment'] = df['review'].apply(analyze_sentiment)

    # Create summary report
    sentiment_summary = df['sentiment'].value_counts().reset_index()
    sentiment_summary.columns = ['sentiment', 'count']

    # Save processed data and summary report to CSV files
    save_to_csv(df, output_file)
    save_to_csv(sentiment_summary, summary_file)


if __name__ == "__main__":
    input_file = 'user_review.csv'
    output_file = 'processed_reviews.csv'
    summary_file = 'sentiment_summary.csv'
    main(input_file, output_file, summary_file)
