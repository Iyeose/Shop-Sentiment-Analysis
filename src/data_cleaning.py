import os
import re
from typing import Self
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

import sys
import logging
from src.data_ingestion import data_ingestion
from config.constant import cleaned_Data

logging.basicConfig(level=logging.INFO)

sentiment_data = data_ingestion()


class DataCleaning:
    def __init__(self):
        self._ensure_nltk()

    def _load_nlp(self) -> "spacy.language.Language":
        for model in ("en_core_web_sm", "xx_ent_wiki_sm"):
            try:
                return spacy.load(model)
            except OSError:
                continue
        nlp_fallback = spacy.blank("xx")
        return nlp_fallback
        
    def _ensure_nltk(self) -> None:
        try:
            _ = stopwords.words("english")
        except LookupError:
            nltk.download("stopwords")

        try:
            word_tokenize("test")
        except LookupError:
            nltk.download("punkt")

        # Newer NLTK versions split tokenizer tables into 'punkt_tab'
        try:
            nltk.data.find("tokenizers/punkt_tab/english/")
        except LookupError:
            try:
                nltk.download("punkt_tab")
            except Exception:
                pass

    def clean_text(self, text: str) -> str:
        """
        Cleans raw text by removing noise.
        Processing Steps:
        1. Convert all characters to lowercase.
        2. Remove URLs and special characters.
        3. Remove extra whitespace.

        This helps the model focus on the actual words instead of punctuation.
        """
        text = str(text).lower()
        # The regex below keeps: letters (a-z), accented letters (À-ÿ), numbers, and spaces.
        # Everything else (emojis, punctuation, symbols) is removed.
        text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def lemmatize(text: str) -> str:
        """
        Groups different forms of a word so they can be analyzed as a single item.
        Example: 'running', 'runs', 'ran' all become 'run'.

        We use spacy (NLP library) to handle this linguistically.
        """
        NLP = Self._load_nlp()
        doc = NLP(text)
        return " ".join(token.lemma_ if token.lemma_ else token.text for token in doc)

    def remove_stopwords(self, text: str) -> str:
        """
        Removes common words (the, is, in) that don't carry much sentiment.
        Focuses the model on the meaningful keywords like 'great', 'bad', or 'broken'.
        """
        tokens = word_tokenize(text)
        sw = set(stopwords.words("english"))
        tokens = [t for t in tokens if t not in sw]
        return " ".join(tokens)


def clean_data(sentiment_data: pd.DataFrame):
    try:
        cleaner = DataCleaning()
        sentiment_data['clean_text'] = sentiment_data['review'].apply(cleaner.clean_text)
        sentiment_data['lemma_text'] = sentiment_data['clean_text'].apply(cleaner.lemmatize)
        sentiment_data['final_text'] = sentiment_data['lemma_text'].apply(cleaner.remove_stopwords)

        # Create the 'Answer Key' for the AI
        sentiment_data['label'] = sentiment_data['rating'].apply(lambda r: 0 if r in (1, 2) else (1 if r == 3 else 2))
        sentiment_data = sentiment_data[["review", "final_text", "label"]]
        logging.info("Data successfully cleaned...")
        sentiment_data.head(5)
        sentiment_data.to_csv(Cleaned_Data)
        return sentiment_data
    except Exception as e:
        logging.error(f"error occurred while cleaning the data {e}")

clean_data(sentiment_data)