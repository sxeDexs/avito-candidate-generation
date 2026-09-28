"""
Avito Services: Candidate Generation Pipeline (Recall@50)
=========================================================
Архитектура решения: Two-Stage Hybrid Retrieval & Pre-Ranking

1. First-Stage Retrieval (L1):
   - Dense Semantic Tower A: deepvk/USER-bge-m3 (1024-dim, специализация на поиске в ру регионе интернета)
   - Dense Semantic Tower B: intfloat/multilingual-e5-base (768-dim, контекстная семантика)
   - Sparse Lexical Tower: BM25 + TF-IDF Query Expansion из кликов
   - Continuous Geo-Affinity: затухание расстояния (scale_km = 45.0) + кликовый граф агломераций

2. Pre-Ranking & Compression (L1.5):
   - Отбор пула Top-200 кандидатов с высоким Recall
   - Переранжирование по точным совпадениям в заголовках (Title Overlap, Exact Phrase)
   - Точный матчинг структурированных фильтров (ParamFilt)
   - Разрешение коллизий в мегаполисах по рейтингу объявления (Quality Boost)
   - Детерминированный отбор финальных Top-50 кандидатов
"""

import os
import re
import random
import time
import pickle
from multiprocessing import Pool
from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import sparse
from nltk.stem.snowball import SnowballStemmer

# ==============================================================================
# КОНФИГУРАЦИЯ И ПАРАМЕТРЫ
# ==============================================================================
SEED = 42
DATA_DIR = Path("dataset")
CACHE_DIR = Path("cache")
OUTPUT_FILE = Path("answer.csv")

# Гиперпараметры первого этапа (L1 Retrieval)
POOL_K = 200               # Размер пула кандидатов для пре-ранкинга
ALPHA = 0.60               # Баланс между BM25 (0.60) и Dense Ensemble (0.40)
W_BGE = 0.60               # Вес USER-bge-m3 в плотном ансамбле
W_E5 = 0.40                # Вес E5-base в плотном ансамбле
SCALE_KM = 45.0            # Масштаб полузатухания расстояния агломераций (км)
GEO_BOOST = 5.0            # Сила мультипликативного гео-буста

# Гиперпараметры пре-ранкинга (L1.5 Filtering)
W_TITLE_OVERLAP = 0.20     # Вес покрытия слов запроса в заголовке
W_PARAMS_OVERLAP = 0.10    # Вес покрытия слов запроса в параметрах
W_EXACT_PHRASE = 0.15      # Бонус за точное вхождение поисковой фразы целиком
MULT_PARAM_FILT = 1.20     # Множитель за совпадение выбранного чекбокса фильтра
NUM_WORKERS = 12           # Потоки CPU для параллельной токенизации


def seed_everything(seed: int = SEED) -> None:
    """Обеспечивает воспроизводимость."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def tokenize_batch(texts: list[str]) -> list[list[str]]:
    """Быстрая токенизация и стемминг батча текстов для BM25."""
    stem = SnowballStemmer("russian").stem
    results = []
    for text in texts:
        if not isinstance(text, str):
            results.append([])
            continue
        tokens = re.findall(r'[a-zа-яё0-9]+', text.lower())
        results.append([stem(token) for token in tokens if len(token) > 1])
    return results


def parallel_tokenize(texts: list[str], n_workers: int = NUM_WORKERS) -> list[list[str]]:
    """Параллельная обработка текстов через пул процессов без утечки памяти."""
    chunk_size = max(1, len(texts) // (n_workers * 4))
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    with Pool(n_workers) as pool:
        results = [token for sub in pool.imap(tokenize_batch, chunks) for token in sub]
    return results


def haversine_np(lat1, lon1, lat2, lon2):
    """
    Векторизованный расчет физического расстояния между координатами в километрах (Haversine Formula).
    Безопасен к границам значений arccos/arcsin (clip).
    """
    lat1 = np.radians(np.asarray(lat1, dtype=np.float64))
    lon1 = np.radians(np.asarray(lon1, dtype=np.float64))
    lat2 = np.radians(np.asarray(lat2, dtype=np.float64))
    lon2 = np.radians(np.asarray(lon2, dtype=np.float64))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    return 2.0 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def build_tfidf_query_expansion(train_df: pd.DataFrame, top_k_words: int = 3) -> dict[str, str]:
    """
    Извлекает скрытые синонимы и профессиональные термины из исторических кликов train.
    Использует TF-IDF взвешивание, исключающее стоп-слова и общую лексику (цена, мастер, выезд).
    """
    print(" -> Построение словаря TF-IDF Query Expansion по обучающей выборке...")
    query_to_tf = defaultdict(Counter)
    global_df = Counter()
    total_docs = len(train_df)

    stop_words = {
        "и", "в", "на", "с", "по", "для", "от", "до", "под", "из", "у", "к", "за", "о", "при",
        "цена", "мастер", "выезд", "гарантия", "рублей", "руб", "услуги", "работа", "недорого"
    }

    for query, title in zip(train_df["search_query"].fillna(""), train_df["item_title_raw"].fillna("")):
        q_norm = str(query).lower().strip()
        if not q_norm:
            continue
        q_tokens = set(re.findall(r'[a-zа-яё0-9]+', q_norm))
        title_tokens = set(re.findall(r'[a-zа-яё0-9]+', str(title).lower()))
        for word in title_tokens:
            if len(word) > 2 and word not in q_tokens and word not in stop_words:
                query_to_tf[q_norm][word] += 1
                global_df[word] += 1

    expansion_dict = {}
    for query, tf_counter in query_to_tf.items():
        scored_terms = [
            (word, count * np.log((total_docs + 1.0) / (global_df[word] + 1.0)))
            for word, count in tf_counter.items()
        ]
        scored_terms.sort(key=lambda item: item[1], reverse=True)
        top_terms = [word for word, _ in scored_terms[:top_k_words]]
        if top_terms:
            expansion_dict[query] = " ".join(top_terms)

    return expansion_dict


class DeterministicBM25:
    """
    Детерминированный BM25 на разреженных матрицах scipy.
    Алфавитная сортировка словаря гарантирует абсолютную повторяемость порядка столбцов.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vocab = {}
        self.doc_matrix = None

    def fit_transform(self, tokenized_docs: list[list[str]]):
        all_words = sorted(set(word for doc in tokenized_docs for word in doc))
        self.vocab = {word: idx for idx, word in enumerate(all_words)}
        vocab_size = len(self.vocab)
        n_docs = len(tokenized_docs)

        rows, cols, data = [], [], []
        df_counts = np.zeros(vocab_size, dtype=np.int32)
        doc_lengths = np.zeros(n_docs, dtype=np.float32)

        for doc_id, doc in enumerate(tokenized_docs):
            doc_lengths[doc_id] = len(doc)
            tf_dict = {}
            for word in doc:
                w_id = self.vocab[word]
                tf_dict[w_id] = tf_dict.get(w_id, 0) + 1
            for w_id, count in tf_dict.items():
                rows.append(doc_id)
                cols.append(w_id)
                data.append(count)
                df_counts[w_id] += 1

        avgdl = np.mean(doc_lengths) if n_docs > 0 else 1.0
        idf_vals = np.log(1.0 + (n_docs - df_counts + 0.5) / (df_counts + 0.5))

        tf_csr = sparse.csr_matrix((data, (rows, cols)), shape=(n_docs, vocab_size), dtype=np.float32)
        len_norm = (self.k1 * (1.0 - self.b + self.b * (doc_lengths / avgdl))).reshape(-1, 1)

        tf_data = tf_csr.data
        tf_rows, tf_cols = tf_csr.nonzero()
        denom = tf_data + len_norm[tf_rows, 0]
        bm25_data = (tf_data * (self.k1 + 1.0)) / denom * idf_vals[tf_cols]

        self.doc_matrix = sparse.csr_matrix((bm25_data, (tf_rows, tf_cols)), shape=(n_docs, vocab_size), dtype=np.float32)
        return self

    def transform_queries(self, tokenized_queries: list[list[str]]) -> sparse.csr_matrix:
        rows, cols, data = [], [], []
        for q_id, q_toks in enumerate(tokenized_queries):
            seen = set()
            for word in q_toks:
                if word in self.vocab and word not in seen:
                    rows.append(q_id)
                    cols.append(self.vocab[word])
                    data.append(1.0)
                    seen.add(word)
        return sparse.csr_matrix((data, (rows, cols)), shape=(len(tokenized_queries), len(self.vocab)), dtype=np.float32)


def main():
    seed_everything(SEED)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("=" * 80)
    print("AVITO SERVICES: CANDIDATE GENERATION PIPELINE")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # 1. ЗАГРУЗКА И ВАЛИДАЦИЯ ДАННЫХ
    # --------------------------------------------------------------------------
    print("\n[1/6] Загрузка исходных Parquet-файлов...")
    train = pd.read_parquet(DATA_DIR / "train.parquet", columns=[
        "search_query", "search_location_id", "item_location_id",
        "item_latitude", "item_longitude", "item_title_raw"
    ]).copy()
    train["item_latitude"] = train["item_latitude"].astype(float)
    train["item_longitude"] = train["item_longitude"].astype(float)

    queries_df = pd.read_parquet(DATA_DIR / "benchmark_queries.parquet")
    items_df = pd.read_parquet(DATA_DIR / "benchmark_items.parquet").copy()
    items_df["item_latitude"] = items_df["item_latitude"].astype(float)
    items_df["item_longitude"] = items_df["item_longitude"].astype(float)

    item_ids = items_df["item_id"].astype(str).values
    query_ids = queries_df["query_id"].astype(str).values
    n_items, n_queries = len(items_df), len(queries_df)

    item_titles = items_df["item_title_raw"].fillna("").astype(str).values
    params = items_df["item_infm_params_text"].fillna("").astype(str).values
    descs = items_df["item_description_raw"].fillna("").astype(str).str.slice(0, 200).values

    item_cats = items_df["item_category_id"].values
    q_locs = queries_df["search_location_id"].values
    q_cats = queries_df["search_category"].values
    raw_queries = queries_df["search_query"].fillna("").astype(str).values
    raw_q_params = queries_df["search_infm_params_text"].fillna("").astype(str).values

    print(f" -> Корпус поиска: {n_items:,} объявлений")
    print(f" -> Тестовых запросов: {n_queries:,}")

    # Мягкий буст проверенных исполнителей (разрешение коллизий одинаковых названий в мегаполисах)
    ratings = items_df["item_rating"].fillna(4.0).values.astype(float)
    reviews = items_df["item_rating_reviews_count"].fillna(0.0).values.astype(float)
    quality_boost = 1.0 + 0.05 * np.clip(ratings / 5.0, 0.5, 1.0) * np.clip(np.log1p(reviews) / np.log1p(100.0), 0.0, 1.0)

    # --------------------------------------------------------------------------
    # 2. ПОСТРОЕНИЕ И КЭШИРОВАНИЕ BM25 С TF-IDF РАСШИРЕНИЕМ
    # --------------------------------------------------------------------------
    print("\n[2/6] Инициализация лексического поиска BM25...")
    bm25_matrix_file = CACHE_DIR / "benchmark_bm25_matrix.npz"
    bm25_vocab_file = CACHE_DIR / "benchmark_bm25_vocab.pkl"

    expansion_dict = build_tfidf_query_expansion(train, top_k_words=3)
    bm25 = DeterministicBM25(k1=1.5, b=0.75)

    if bm25_matrix_file.exists() and bm25_vocab_file.exists():
        print(f" -> Загрузка готовой матрицы BM25 из кэша: {bm25_matrix_file}...")
        bm25.doc_matrix = sparse.load_npz(bm25_matrix_file)
        with open(bm25_vocab_file, "rb") as f:
            bm25.vocab = pickle.load(f)
    else:
        print(" -> Расчет матрицы BM25 на 12 потоках CPU...")
        corpus_bm25 = (item_titles + " " + item_titles + " " + item_titles + " " + params + " " + params + " " + descs).tolist()
        tokenized_docs = parallel_tokenize(corpus_bm25, n_workers=NUM_WORKERS)
        bm25.fit_transform(tokenized_docs)
        sparse.save_npz(bm25_matrix_file, bm25.doc_matrix)
        with open(bm25_vocab_file, "wb") as f:
            pickle.dump(bm25.vocab, f)

    expanded_queries = [
        f"{str(q).lower()} {expansion_dict.get(str(q).lower().strip(), '')} {str(p)}".strip()
        for q, p in zip(queries_df["search_query"], queries_df["search_infm_params_text"])
    ]
    tokenized_queries = parallel_tokenize(expanded_queries, n_workers=NUM_WORKERS)
    bm25_scores = (bm25.transform_queries(tokenized_queries) * bm25.doc_matrix.T.tocsc()).toarray()
    bm25_norm = bm25_scores / np.maximum(bm25_scores.max(axis=1, keepdims=True), 1e-6)

    # --------------------------------------------------------------------------
    # 3. ПОСТРОЕНИЕ МАТРИЦЫ SMOOTH GEO-AFFINITY (scale_km = 45.0)
    # --------------------------------------------------------------------------
    print("\n[3/6] Построение непрерывного гео-пространства агломераций...")
    all_coords = pd.concat([
        train[["item_location_id", "item_latitude", "item_longitude"]],
        items_df[["item_location_id", "item_latitude", "item_longitude"]]
    ]).dropna(subset=["item_latitude", "item_longitude"])

    search_coords = train.dropna(subset=["item_latitude", "item_longitude"]).groupby("search_location_id")[["item_latitude", "item_longitude"]].median()
    item_coords = all_coords.groupby("item_location_id")[["item_latitude", "item_longitude"]].median()

    loc_pairs = train.groupby(["search_location_id", "item_location_id"]).size()
    loc_totals = train.groupby("search_location_id").size()
    cooccur_probs = (loc_pairs / loc_totals).to_dict()

    unique_q_locs = np.array(sorted(queries_df["search_location_id"].unique()))
    unique_it_locs = np.array(sorted(items_df["item_location_id"].unique()))

    q_to_idx = {loc: idx for idx, loc in enumerate(unique_q_locs)}
    it_to_idx = {loc: idx for idx, loc in enumerate(unique_it_locs)}
    affinity = np.zeros((len(unique_q_locs), len(unique_it_locs)), dtype=np.float32)

    # Точные совпадения ID локаций
    for loc, q_idx in q_to_idx.items():
        if loc in it_to_idx:
            affinity[q_idx, it_to_idx[loc]] = 1.0

    # Эмпирические клики из train
    for (ql, itl), prob in cooccur_probs.items():
        if ql in q_to_idx and itl in it_to_idx:
            affinity[q_to_idx[ql], it_to_idx[itl]] = max(affinity[q_to_idx[ql], it_to_idx[itl]], min(1.0, float(prob) * 1.2))

    # Рациональное затухание по реальным координатам
    common_q = [loc for loc in unique_q_locs if loc in search_coords.index]
    common_it = [loc for loc in unique_it_locs if loc in item_coords.index]
    if common_q and common_it:
        dists = haversine_np(
            search_coords.loc[common_q, "item_latitude"].values[:, None],
            search_coords.loc[common_q, "item_longitude"].values[:, None],
            item_coords.loc[common_it, "item_latitude"].values[None, :],
            item_coords.loc[common_it, "item_longitude"].values[None, :]
        )
        smooth_weights = 1.0 / (1.0 + (dists / SCALE_KM))
        for i, ql in enumerate(common_q):
            qi = q_to_idx[ql]
            for j, itl in enumerate(common_it):
                affinity[qi, it_to_idx[itl]] = max(affinity[qi, it_to_idx[itl]], smooth_weights[i, j])

    it_loc_indices = np.array([it_to_idx[loc] for loc in items_df["item_location_id"].values])

    # --------------------------------------------------------------------------
    # 4. СЕМАНТИЧЕСКИЙ АНСАМБЛЬ (E5-BASE + USER-BGE-M3)
    # --------------------------------------------------------------------------
    print("\n[4/6] Dense Retrieval: двухбашенный ансамбль E5-Base + USER-BGE-M3...")
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True

    # Общий текст для кодирования объявлений (если кэша нет)
    it_dense_texts = (item_titles + ". " + item_titles + ". " + pd.Series(params).str.slice(0, 150) + ". " + descs).tolist()

    # ==================== БАШНЯ А: intfloat/multilingual-e5-base ====================
    cache_e5 = CACHE_DIR / "benchmark_embeddings_e5.pt"
    if not cache_e5.exists() and Path("benchmark_embeddings_e5.pt").exists():
        cache_e5 = Path("benchmark_embeddings_e5.pt")

    model_e5 = SentenceTransformer("intfloat/multilingual-e5-base", device=device)

    if cache_e5.exists():
        print(f" -> [E5-Base] Загрузка готовых векторов из кэша: {cache_e5}...")
        it_embeddings_e5 = torch.load(cache_e5, weights_only=True).to(device)
    else:
        print(" -> [E5-Base] Кэш не найден. Кодирование корпуса на GPU...")
        passage_texts_e5 = ["passage: " + t for t in it_dense_texts]
        with torch.inference_mode():
            it_embeddings_e5 = model_e5.encode(
                passage_texts_e5, batch_size=256, show_progress_bar=True,
                normalize_embeddings=True, convert_to_tensor=True, device=device
            ).half()
        torch.save(it_embeddings_e5.cpu(), CACHE_DIR / "benchmark_embeddings_e5.pt")
        it_embeddings_e5 = it_embeddings_e5.to(device)

    print(" -> [E5-Base] Кодирование тестовых запросов...")
    q_texts_e5 = ["query: " + (str(q or "") + " " + str(p or "")) for q, p in zip(queries_df["search_query"], queries_df["search_infm_params_text"])]
    with torch.inference_mode():
        q_emb_e5 = model_e5.encode(q_texts_e5, batch_size=256, normalize_embeddings=True, convert_to_tensor=True, device=device).half()
        dense_scores_e5 = torch.matmul(q_emb_e5, it_embeddings_e5.T).cpu().numpy()

    del model_e5, it_embeddings_e5, q_emb_e5
    torch.cuda.empty_cache()
    dense_norm_e5 = np.clip((dense_scores_e5 - 0.70) / np.maximum(dense_scores_e5.max(axis=1, keepdims=True) - 0.70, 1e-6), 0.0, 1.0)

    # ==================== БАШНЯ Б: deepvk/USER-bge-m3 ====================
    cache_bge = CACHE_DIR / "benchmark_embeddings_bge_m3.pt"
    if not cache_bge.exists() and Path("benchmark_embeddings_bge_m3.pt").exists():
        cache_bge = Path("benchmark_embeddings_bge_m3.pt")

    model_bge = SentenceTransformer("deepvk/USER-bge-m3", device=device)

    if cache_bge.exists():
        print(f" -> [USER-BGE-M3] Загрузка готовых векторов из кэша: {cache_bge}...")
        it_embeddings_bge = torch.load(cache_bge, weights_only=True).to(device)
    else:
        print(" -> [USER-BGE-M3] Кэш не найден. Кодирование корпуса на GPU...")
        with torch.inference_mode():
            it_embeddings_bge = model_bge.encode(
                it_dense_texts, batch_size=128, show_progress_bar=True,
                normalize_embeddings=True, convert_to_tensor=True, device=device
            ).half()
        torch.save(it_embeddings_bge.cpu(), CACHE_DIR / "benchmark_embeddings_bge_m3.pt")
        it_embeddings_bge = it_embeddings_bge.to(device)

    print(" -> [USER-BGE-M3] Кодирование тестовых запросов...")
    q_texts_bge = [(str(q or "") + " " + str(p or "")).strip() for q, p in zip(queries_df["search_query"], queries_df["search_infm_params_text"])]
    with torch.inference_mode():
        q_emb_bge = model_bge.encode(q_texts_bge, batch_size=256, normalize_embeddings=True, convert_to_tensor=True, device=device).half()
        dense_scores_bge = torch.matmul(q_emb_bge, it_embeddings_bge.T).cpu().numpy()

    del model_bge, it_embeddings_bge, q_emb_bge
    torch.cuda.empty_cache()
    dense_norm_bge = (dense_scores_bge - dense_scores_bge.min(axis=1, keepdims=True)) / np.maximum(dense_scores_bge.max(axis=1, keepdims=True) - dense_scores_bge.min(axis=1, keepdims=True), 1e-6)

    # Взвешенное объединение плотных представлений (0.6 BGE + 0.4 E5)
    print(" -> Слияние плотных представлений (0.6 BGE + 0.4 E5)...")
    dense_norm = W_BGE * dense_norm_bge + W_E5 * dense_norm_e5

    # --------------------------------------------------------------------------
    # 5. ПРЕ-РАНКИНГ И СЖАТИЕ КАНДИДАТОВ (TOP-200 -> TOP-50)
    # --------------------------------------------------------------------------
    print("\n[5/6] Пре-ранкинг L1.5: Title Overlap + ParamFilt + Quality Boost...")
    item_titles_lower = [t.lower() for t in item_titles]
    item_title_words = [set(re.findall(r'[a-zа-яё0-9]+', t)) for t in item_titles_lower]
    item_params_lower = [str(p).lower() for p in params]
    item_param_words = [set(re.findall(r'[a-zа-яё0-9]+', p)) for p in item_params_lower]

    base_scores = ALPHA * bm25_norm + (1.0 - ALPHA) * dense_norm
    is_cat_114 = (item_cats == 114)
    predictions = []

    for i in range(n_queries):
        score = base_scores[i].copy()

        # Приоритет категории услуг 114 (активен для категорий 114 и 0)
        if q_cats[i] in [114, 0]:
            score *= np.where(is_cat_114, 1.0, 0.05)

        # Мультипликативный непрерывный гео-буст
        aff = affinity[q_to_idx[q_locs[i]], it_loc_indices] if q_locs[i] in q_to_idx else np.zeros(n_items, dtype=np.float32)
        score *= (1.0 + aff * GEO_BOOST)

        # Выделение широкого пула кандидатов Top-200
        top_k_idx = np.argpartition(score, -POOL_K)[-POOL_K:]

        q_raw = raw_queries[i].lower().strip()
        q_words = set(re.findall(r'[a-zа-яё0-9]+', q_raw))
        q_len = max(len(q_words), 1)
        q_filter_str = raw_q_params[i].lower().strip()

        has_local = np.any(aff[top_k_idx] >= 0.5)

        # Пре-ранкинг внутри пула Топ-200
        reranked_scores = []
        for cand_idx in top_k_idx:
            s = score[cand_idx]
            t_words = item_title_words[cand_idx]
            p_words = item_param_words[cand_idx]

            t_overlap = len(q_words & t_words) / q_len
            p_overlap = len(q_words & p_words) / q_len
            exact_phrase = 1.0 if q_raw and (q_raw in item_titles_lower[cand_idx]) else 0.0

            bonus = 1.0 + W_TITLE_OVERLAP * t_overlap + W_PARAMS_OVERLAP * p_overlap + W_EXACT_PHRASE * exact_phrase

            # ParamFilt: прямое совпадение выбранного чекбокса фильтра в параметрах объявления
            if q_filter_str and (q_filter_str in item_params_lower[cand_idx]):
                bonus *= MULT_PARAM_FILT

            # Local Shield: защита от далеких городов (>140 км), если в городе есть свои мастера
            if has_local and aff[cand_idx] < 0.15:
                bonus *= 0.85

            # Учет социального доверия (рейтинг и отзывы) при равных заголовках
            bonus *= quality_boost[cand_idx]
            reranked_scores.append(s * bonus)

        reranked_scores = np.array(reranked_scores)
        best50_local = np.argpartition(reranked_scores, -50)[-50:]

        # Двухуровневый детерминированный tie-breaking (скор убывает, ID возрастает)
        tie_keys = item_ids[top_k_idx[best50_local]]
        score_keys = -reranked_scores[best50_local]
        sorted_order = np.lexsort((tie_keys, score_keys))

        final_idx = top_k_idx[best50_local][sorted_order]
        predictions.append(" ".join(item_ids[final_idx]))

    # --------------------------------------------------------------------------
    # 6. ФОРМИРОВАНИЕ И СОХРАНЕНИЕ ANSWER.CSV
    # --------------------------------------------------------------------------
    print("\n[6/6] Сохранение финального файла answer.csv...")
    sub = pd.DataFrame({"query_id": query_ids, "answer": predictions})
    sub.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\nВремя работы пайплайна: {time.time() - t0:.1f} сек.")
    print("Файл answer.csv готов.")


if __name__ == "__main__":
    main()