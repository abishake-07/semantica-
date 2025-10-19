import pandas as pd
from typing import List, Dict


def load_wands_csv(path: str, n_rows: int = None) -> List[Dict]:
    """Load Wayfair Annotation Dataset (WANDS) style CSV and return list of product dicts.

    Expected columns: product_id, title, description, attributes (optional)
    """
    df = pd.read_csv(path, nrows=n_rows)
    products = []
    for _, row in df.iterrows():
        products.append({
            "product_id": str(row.get("product_id", "")),
            "title": str(row.get("title", "")),
            "description": str(row.get("description", "")),
            "attributes": row.get("attributes", None),
        })
    return products
