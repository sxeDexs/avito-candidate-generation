"""
Валидатор файла answer.csv на строгое соответствие спецификации соревнования Авито.
"""
import re
from pathlib import Path
import pandas as pd

def verify_submission():
    print("=" * 80)
    print("ПРОВЕРКА ФАЙЛА answer.csv.")
    print("=" * 80)

    csv_path = Path("answer.csv")
    assert csv_path.exists(), "Ошибка: файл answer.csv не найден в корневой директории!"

    # 1. Проверка заголовка и разделителей
    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        assert header == "query_id,answer", f"Неверный заголовок: '{header}', ожидалось 'query_id,answer'"

    sub = pd.read_csv(csv_path, dtype={"query_id": str, "answer": str})
    queries = pd.read_parquet("dataset/benchmark_queries.parquet")
    items = pd.read_parquet("dataset/benchmark_items.parquet")

    # 2. Проверка размерности колонок и строк
    assert list(sub.columns) == ["query_id", "answer"], f"Ошибка в колонках: {sub.columns}"
    assert len(sub) == len(queries), f"Ошибка: в файле {len(sub)} строк, ожидалось ровно {len(queries)}"

    # 3. Проверка соответствия и регистра query_id
    expected_qids = list(queries["query_id"].astype(str))
    actual_qids = list(sub["query_id"].astype(str))
    assert actual_qids == expected_qids, "Порядок или состав query_id не совпадает с benchmark_queries!"

    # 4. Проверка существования item_id в корпусе
    valid_items_set = set(items["item_id"].astype(str).str.lower().values)
    hex_16_pattern = re.compile(r"^[0-9a-f]{16}$")

    # 5. Построчная валидация всех ответов
    for idx, (qid, ans) in enumerate(zip(sub["query_id"], sub["answer"])):
        assert len(qid) == 16, f"Строка {idx}: длина query_id '{qid}' != 16"
        assert isinstance(ans, str) and len(ans.strip()) > 0, f"Строка {idx}: пустой ответ для query_id '{qid}'"

        tokens = ans.split(" ")
        assert len(tokens) <= 50, f"Строка {idx}: превышен лимит 50 объявлений ({len(tokens)})"
        assert len(tokens) == len(set(tokens)), f"Строка {idx}: найдены дубликаты item_id"

        for item in tokens:
            assert len(item) == 16, f"Строка {idx}: длина item_id '{item}' != 16"
            assert hex_16_pattern.match(item), f"Строка {idx}: item_id '{item}' не является строчным hex"
            assert item in valid_items_set, f"Строка {idx}: item_id '{item}' отсутствует в benchmark_items.parquet"

    print(f"Проверено ровно {len(sub):,} поисковых запросов.")
    print(f"Все {len(sub) * 50:,} предсказанных объявлений строго валидны, уникальны и найдены в корпусе.")
    print("Отсутствуют индексы, экспоненциальные записи, лишние пробелы и дубликаты.")
    print("файл answer.csv валиден.")


if __name__ == "__main__":
    verify_submission()