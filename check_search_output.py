import csv
import json
from pathlib import Path

from search_core import discover_product_files


DATA_DIR = Path("data")
OUTPUT_PATH = DATA_DIR / "search_output_top_20.json"
EXPECTED_PATH = DATA_DIR / "expected_ids_50.json"

ID_COLUMN_NAMES = ["id", "ID", "Id", "product_id", "Product ID", "productId"]


def detect_csv_delimiter(path):
    sample = path.read_text(
        encoding="utf-8-sig",
        errors="replace",
    )[:4096]

    try:
        dialect = csv.Sniffer().sniff(
            sample,
            delimiters=["\t", ";", ",", "|"],
        )
        return dialect.delimiter
    except Exception:
        if "\t" in sample:
            return "\t"

        return ";"


def field_value(value):
    if value is None:
        return ""

    value = str(value).strip()

    if not value or value == "-":
        return ""

    return value


def first_existing_value(row, column_names):
    for column_name in column_names:
        if column_name in row and field_value(row.get(column_name)):
            return field_value(row.get(column_name))

    return ""


def load_product_ids():
    ids = set()
    product_files = discover_product_files(DATA_DIR)

    for path in product_files:
        path = Path(path)

        if not path.exists() or path.suffix.lower() != ".csv":
            continue

        delimiter = detect_csv_delimiter(path)

        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file, delimiter=delimiter)

            for row_number, row in enumerate(reader, start=1):
                product_id = first_existing_value(row, ID_COLUMN_NAMES)

                if product_id:
                    ids.add(str(product_id))

    return ids


def main():
    with OUTPUT_PATH.open("r", encoding="utf-8") as file:
        output = json.load(file)

    product_ids = load_product_ids()

    format_errors = []
    unknown_ids = []

    for row in output:
        if "query_id" not in row:
            format_errors.append("missing query_id")

        if "query_text" not in row:
            format_errors.append(f"{row.get('query_id')}: missing query_text")

        if "top_20_product_ids" not in row:
            format_errors.append(f"{row.get('query_id')}: missing top_20_product_ids")
            continue

        if not isinstance(row["top_20_product_ids"], list):
            format_errors.append(f"{row.get('query_id')}: top_20_product_ids is not list")
            continue

        if len(row["top_20_product_ids"]) > 20:
            format_errors.append(f"{row.get('query_id')}: more than 20 ids")

        for product_id in row["top_20_product_ids"]:
            if str(product_id) not in product_ids:
                unknown_ids.append((row.get("query_id"), product_id))

    print("Rows:", len(output))
    print("Known product ids:", len(product_ids))
    print("Format errors:", len(format_errors))
    print("Unknown ids:", len(unknown_ids))

    if format_errors:
        print("First format errors:")
        print(format_errors[:10])

    if unknown_ids:
        print("First unknown ids:")
        print(unknown_ids[:10])

    if EXPECTED_PATH.exists():
        with EXPECTED_PATH.open("r", encoding="utf-8") as file:
            expected = json.load(file)

        output_by_qid = {
            row["query_id"]: set(str(x) for x in row["top_20_product_ids"])
            for row in output
        }

        hits = 0
        total = 0

        for row in expected:
            query_id = row["query_id"]
            expected_ids = set(str(x) for x in row["expected_relevant_product_ids"])
            predicted_ids = output_by_qid.get(query_id, set())

            total += 1

            if expected_ids & predicted_ids:
                hits += 1

        print("Hit@20:", hits / total if total else 0)
        print("Misses:", total - hits)

    if not format_errors and not unknown_ids:
        print("OK: output format is correct")


if __name__ == "__main__":
    main()
