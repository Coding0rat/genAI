"""Cartwise: grounded shopping recommendations from a local product CSV."""

from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
CATALOG_PATH = BASE_DIR / "uploaded_catalog" / "flipkart_com-ecommerce_sample.csv"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-6.2-astra")
SHORTLIST_SIZE = 18

REQUIRED_COLUMNS = [
    "pid",
    "product_name",
    "product_category_tree",
    "retail_price",
    "discounted_price",
    "image",
    "description",
    "product_rating",
    "brand",
    "product_url",
]


def clean_text(value: Any, limit: int | None = None) -> str:
    """Turn a CSV field into compact display/prompt text."""
    if pd.isna(value):
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit].rstrip() if limit else text


def category_path(value: Any) -> str:
    """Extract the first category hierarchy from Flipkart's JSON-like column."""
    raw = clean_text(value)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and parsed:
            raw = str(parsed[0])
    except (json.JSONDecodeError, TypeError):
        raw = raw.strip("[]\"")
    return clean_text(raw)


def first_image(value: Any) -> str:
    """Return the first product image URL, if a valid JSON list is present."""
    try:
        images = json.loads(clean_text(value))
        if isinstance(images, list) and images:
            return str(images[0])
    except (json.JSONDecodeError, TypeError):
        pass
    return ""


@st.cache_data(show_spinner="Preparing the product catalog…")
def load_catalog(csv_path: str) -> pd.DataFrame:
    """Load and normalize only the fields used by the recommender."""
    frame = pd.read_csv(
        csv_path,
        usecols=REQUIRED_COLUMNS,
        low_memory=False,
        encoding_errors="replace",
        on_bad_lines="skip",
    )
    frame = frame.dropna(subset=["product_name"]).copy()
    frame["product_name"] = frame["product_name"].map(lambda value: clean_text(value, 140))
    frame["brand"] = frame["brand"].map(lambda value: clean_text(value, 60) or "Unbranded")
    frame["product_url"] = frame["product_url"].map(lambda value: clean_text(value, 500))
    frame["category_path"] = frame["product_category_tree"].map(category_path)
    frame["category"] = frame["category_path"].str.split(" >> ").str[0].fillna("Other")
    frame["description_short"] = frame["description"].map(lambda value: clean_text(value, 420))
    frame["image_url"] = frame["image"].map(first_image)
    frame["price"] = pd.to_numeric(frame["discounted_price"], errors="coerce")
    retail = pd.to_numeric(frame["retail_price"], errors="coerce")
    frame["price"] = frame["price"].fillna(retail)
    frame["retail_price"] = retail
    frame["rating"] = pd.to_numeric(frame["product_rating"], errors="coerce")
    frame["discount_percent"] = (
        ((retail - frame["price"]) / retail.replace(0, pd.NA)) * 100
    ).clip(lower=0, upper=100).fillna(0)
    frame = frame[frame["price"].notna() & (frame["price"] > 0)].copy()
    frame["catalog_id"] = frame.index.astype(str)
    frame["search_text"] = (
        frame["product_name"].str.lower()
        + " "
        + frame["brand"].str.lower()
        + " "
        + frame["category_path"].str.lower()
        + " "
        + frame["description_short"].str.lower()
    )
    return frame.reset_index(drop=True)


def search_products(
    catalog: pd.DataFrame,
    category: str,
    max_budget: int,
    needs: str,
    sort_by: str,
) -> pd.DataFrame:
    """Apply hard filters, then create a relevance-ranked shortlist locally."""
    products = catalog[catalog["price"] <= max_budget].copy()
    if category != "All categories":
        products = products[products["category"] == category].copy()

    needs = clean_text(needs).lower()
    tokens = list(dict.fromkeys(re.findall(r"[a-z0-9]{2,}", needs)))
    products["match_score"] = 0.0
    for token in tokens:
        escaped = re.escape(token)
        products["match_score"] += products["product_name"].str.count(escaped) * 9
        products["match_score"] += products["category_path"].str.count(escaped) * 5
        products["match_score"] += products["description_short"].str.lower().str.count(escaped) * 2

    # Quality/value signals only decide ordering; category and price always remain hard rules.
    products["match_score"] += products["rating"].fillna(0) * 0.75
    products["match_score"] += products["discount_percent"] / 25
    products = products.drop_duplicates(subset=["pid"], keep="first")

    ordering = {
        "Best match": ["match_score", "rating", "discount_percent", "price"],
        "Lowest price": ["price", "match_score", "rating"],
        "Highest rated": ["rating", "match_score", "price"],
        "Biggest discount": ["discount_percent", "match_score", "rating"],
    }
    ascending = {
        "Best match": [False, False, False, True],
        "Lowest price": [True, False, False],
        "Highest rated": [False, False, True],
        "Biggest discount": [False, False, False],
    }
    return products.sort_values(ordering[sort_by], ascending=ascending[sort_by], na_position="last")


def candidate_payload(products: pd.DataFrame) -> list[dict[str, Any]]:
    """Build compact, factual records for the model—no unfiltered catalog data."""
    fields = [
        "catalog_id",
        "product_name",
        "brand",
        "category_path",
        "price",
        "retail_price",
        "rating",
        "discount_percent",
        "description_short",
    ]
    records = products.head(SHORTLIST_SIZE)[fields].fillna("").to_dict(orient="records")
    for record in records:
        record["price"] = round(float(record["price"]), 2)
        record["retail_price"] = (
            round(float(record["retail_price"]), 2) if record["retail_price"] != "" else None
        )
        record["rating"] = round(float(record["rating"]), 1) if record["rating"] != "" else None
        record["discount_percent"] = round(float(record["discount_percent"]), 1)
    return records


def extract_json(text: str) -> dict[str, Any]:
    """Accept plain JSON or a JSON code block while avoiding unsafe parsing."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("The model did not return a JSON object.")
    data = json.loads(cleaned[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("The model response was not an object.")
    return data


def get_llm_recommendations(
    products: pd.DataFrame, needs: str, max_budget: int, model: str
) -> tuple[list[dict[str, str]], str]:
    """Rank only approved candidate rows with the Responses API and validate IDs."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("No OPENAI_API_KEY is configured.")

    # Import lazily so the local fallback can still run when dependencies are incomplete.
    from openai import OpenAI

    payload = candidate_payload(products)
    prompt = f"""Shopper needs: {clean_text(needs, 600) or "No extra preferences stated."}
Maximum budget: INR {max_budget:,.0f}

Choose exactly three distinct products from the candidate JSON below. Do not invent
products, prices, specifications, ratings, or IDs. Use only catalog_id values that
appear in the JSON. Give concrete, modest reasons that connect the stated needs to
the provided fields. A missing rating is not a positive signal.

Return JSON only in this exact shape:
{{"recommendations":[{{"catalog_id":"string","reason":"one or two sentences"}}],"summary":"one concise sentence"}}

Candidates:
{json.dumps(payload, ensure_ascii=False)}"""

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model.strip() or DEFAULT_MODEL,
        instructions=(
            "You are a careful shopping assistant. Your job is to rank the supplied "
            "catalog records, not to browse or use outside knowledge."
        ),
        input=prompt,
    )
    response_data = extract_json(response.output_text)
    valid_ids = set(products.head(SHORTLIST_SIZE)["catalog_id"].astype(str))
    recommendations: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in response_data.get("recommendations", []):
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("catalog_id", ""))
        reason = clean_text(item.get("reason", ""), 360)
        if item_id in valid_ids and item_id not in seen and reason:
            recommendations.append({"catalog_id": item_id, "reason": reason})
            seen.add(item_id)
        if len(recommendations) == 3:
            break
    if len(recommendations) < 3:
        raise ValueError("The model did not select three valid catalog products.")
    return recommendations, clean_text(response_data.get("summary", ""), 240)


def local_recommendations(products: pd.DataFrame, needs: str) -> tuple[list[dict[str, str]], str]:
    """Explain the top local matches when OpenAI is intentionally unavailable."""
    recommendations: list[dict[str, str]] = []
    for _, product in products.head(3).iterrows():
        details: list[str] = []
        if pd.notna(product["rating"]):
            details.append(f"rated {product['rating']:.1f}/5")
        if product["discount_percent"] >= 5:
            details.append(f"{product['discount_percent']:.0f}% below listed retail price")
        if needs.strip():
            details.append("matches the locally filtered category, budget, and keyword shortlist")
        else:
            details.append("is a strong value within the selected category and budget")
        recommendations.append(
            {"catalog_id": str(product["catalog_id"]), "reason": "; ".join(details).capitalize() + "."}
        )
    return recommendations, "Ranked locally from your filtered catalog; add an API key for AI-written comparisons."


def render_product_card(product: pd.Series, rank: int, reason: str) -> None:
    """Render one grounded recommendation with source catalog details."""
    with st.container(border=True):
        image_col, content_col = st.columns([1, 4], vertical_alignment="top")
        with image_col:
            if product["image_url"]:
                st.image(product["image_url"], use_container_width=True)
            else:
                st.markdown("### 🛍️")
        with content_col:
            st.markdown(f"#### {rank}. {product['product_name']}")
            st.caption(f"{product['brand']} · {product['category_path']}")
            price_line = f"**₹{product['price']:,.0f}**"
            if pd.notna(product["retail_price"]) and product["retail_price"] > product["price"]:
                price_line += f"  ~~₹{product['retail_price']:,.0f}~~"
            if pd.notna(product["rating"]):
                price_line += f"  ·  ⭐ {product['rating']:.1f}/5"
            st.markdown(price_line)
            st.write(reason)
            if product["product_url"]:
                st.link_button("View catalog link", product["product_url"])


def main() -> None:
    load_dotenv(BASE_DIR / ".env")
    st.set_page_config(page_title="Cartwise | Shopping Assistant", page_icon="🛒", layout="wide")
    st.title("Cartwise 🛒")
    st.caption("Tell us what you need. We filter the catalog first, then explain the best three choices.")

    if not CATALOG_PATH.exists():
        st.error(f"Catalog not found at `{CATALOG_PATH.relative_to(BASE_DIR)}`.")
        st.stop()

    try:
        catalog = load_catalog(str(CATALOG_PATH))
    except Exception as exc:
        st.error(f"Could not read the product catalog: {exc}")
        st.stop()

    categories = ["All categories", *sorted(catalog["category"].dropna().unique())]
    max_catalog_price = int(catalog["price"].quantile(0.99))
    max_catalog_price = max(1_000, min(max_catalog_price, 200_000))

    with st.sidebar:
        st.header("Your filters")
        selected_category = st.selectbox("Category", categories)
        budget = st.number_input("Maximum budget (₹)", min_value=1, value=min(5_000, max_catalog_price), step=500)
        sort_by = st.selectbox("Prioritize", ["Best match", "Lowest price", "Highest rated", "Biggest discount"])
        use_ai = st.toggle("Use OpenAI for the final ranking", value=bool(os.getenv("OPENAI_API_KEY")))
        model = st.text_input("OpenAI model", value=DEFAULT_MODEL, disabled=not use_ai)
        st.divider()
        st.caption(f"Catalog: {len(catalog):,} products · historical INR prices")

    needs = st.text_area(
        "What are you looking for?",
        placeholder="Example: comfortable casual shoes for daily use, preferably durable and well rated",
        height=110,
    )
    find_clicked = st.button("Find my top 3", type="primary", use_container_width=True)

    if not find_clicked:
        st.info("Set a budget and describe your needs, then choose **Find my top 3**.")
        return

    filtered = search_products(catalog, selected_category, int(budget), needs, sort_by)
    if filtered.empty:
        st.warning("No products match that category and budget. Try raising the budget or selecting all categories.")
        return

    st.caption(f"{len(filtered):,} products passed the hard filters. Ranking the best {min(SHORTLIST_SIZE, len(filtered)):,} candidates.")
    source = "local ranking"
    summary = ""
    if use_ai:
        try:
            with st.spinner("Comparing the shortlisted products…"):
                recommendation_data, summary = get_llm_recommendations(filtered, needs, int(budget), model)
            source = f"OpenAI ({model.strip() or DEFAULT_MODEL})"
        except Exception as exc:
            st.warning(f"AI ranking was unavailable ({clean_text(exc, 180)}). Showing the local ranking instead.")
            recommendation_data, summary = local_recommendations(filtered, needs)
    else:
        recommendation_data, summary = local_recommendations(filtered, needs)

    product_by_id = filtered.set_index("catalog_id", drop=False)
    st.subheader("Your three best options")
    if summary:
        st.info(summary)
    for rank, item in enumerate(recommendation_data, start=1):
        if item["catalog_id"] in product_by_id.index:
            render_product_card(product_by_id.loc[item["catalog_id"]], rank, item["reason"])

    with st.expander("See the filtered shortlist"):
        display = filtered.head(SHORTLIST_SIZE)[
            ["product_name", "brand", "category", "price", "rating", "discount_percent"]
        ].rename(columns={"product_name": "Product", "brand": "Brand", "category": "Category", "price": "Price (₹)", "rating": "Rating", "discount_percent": "Discount (%)"})
        st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption(f"Recommendation source: {source}. Prices and product data are from the supplied historical CSV.")


if __name__ == "__main__":
    main()
