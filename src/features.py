from __future__ import annotations

import re
from html import unescape
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


def clean_text(value: str) -> str:
    if not isinstance(value, str):
        value = "" if pd.isna(value) else str(value)
    text = unescape(value)
    text = TAG_RE.sub(" ", text)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def compose_text_features(df: pd.DataFrame) -> pd.Series:
    title = df["title"].fillna("").map(clean_text)
    text = df["text"].fillna("").map(clean_text)
    source = df["source"].fillna("").astype(str)
    publication_date = pd.to_datetime(df["publication_date"], errors="coerce")
    year = publication_date.dt.year.fillna(-1).astype(int).astype(str)
    month = publication_date.dt.month.fillna(-1).astype(int).astype(str)

    composed = (
        "[SRC] " + source
        + " [YEAR] " + year
        + " [MONTH] " + month
        + " [TITLE] " + title
        + " [TEXT] " + text
    )
    return composed


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def build_transformer_embeddings(
    texts: pd.Series,
    model_name: str = "intfloat/multilingual-e5-small",
    batch_size: int = 32,
    max_length: int = 256,
    cache_path: str | Path | None = None,
) -> np.ndarray:
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            loaded = np.load(cache_path)
            return loaded["embeddings"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings: list[np.ndarray] = []
    use_amp = device == "cuda"

    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), batch_size), desc=f"Embeddings:{model_name}"):
            batch = texts.iloc[start:start + batch_size].tolist()
            batch = [f"passage: {x}" for x in batch]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            with torch.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(**encoded)
                pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

            all_embeddings.append(pooled.float().cpu().numpy())

    embeddings = np.vstack(all_embeddings)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, embeddings=embeddings)
    return embeddings


def build_tfidf_features(
    train_texts: pd.Series,
    test_texts: pd.Series,
    max_features_word: int = 120000,
    max_features_char: int = 80000,
):
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.98,
        sublinear_tf=True,
        max_features=max_features_word,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_df=0.98,
        sublinear_tf=True,
        max_features=max_features_char,
    )

    x_train_word = word_vectorizer.fit_transform(train_texts)
    x_test_word = word_vectorizer.transform(test_texts)
    x_train_char = char_vectorizer.fit_transform(train_texts)
    x_test_char = char_vectorizer.transform(test_texts)

    x_train = hstack([x_train_word, x_train_char]).tocsr()
    x_test = hstack([x_test_word, x_test_char]).tocsr()
    return x_train, x_test
