
import gzip
import json

file_path = "data/meta_All_Beauty.jsonl.gz"

with gzip.open(file_path, "rt", encoding="utf-8") as file:
    for i, line in enumerate(file):
        product = json.loads(line)

        print("Product number:", i + 1)
        print("Product ID:", product.get("parent_asin"))
        print("Product name:", product.get("title"))
        print("-" * 40)

        if i == 4:
            break
