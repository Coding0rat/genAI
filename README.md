# Cartwise — Shopping Assistant

Cartwise searches the supplied Flipkart product CSV locally, applying the shopper's category, price cap, and stated needs before it sends a compact candidate shortlist to OpenAI. The model then returns three grounded recommendations with reasons. If no API key is configured, the app still works with a transparent local ranking fallback.

## Setup

1. Keep the extracted catalog at `uploaded_catalog/flipkart_com-ecommerce_sample.csv`. It is already in place for this project.
2. Create and activate a virtual environment, then install the dependencies:

   ```powershell
   uv venv
   .\.venv\Scripts\Activate.ps1
   uv pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env`, add `OPENAI_API_KEY`, and optionally change `OPENAI_MODEL` to a model available to your OpenAI project. The key stays on the server running Streamlit and is not exposed in the browser.
4. Start the app:

   ```powershell
   streamlit run app.py
   ```

## How it works

1. **Pandas filtering:** Reads only the useful CSV columns, normalizes category paths and prices, and filters by root category and maximum price.
2. **Local shortlist:** Scores matches against the shopper's needs using product name, category, description, ratings, and discount. Only the best 18 products are candidates.
3. **LLM ranking:** Sends those compact rows—never the full catalog—to the OpenAI Responses API. The prompt restricts the model to the supplied IDs and asks for JSON containing exactly three concise reasons.
4. **Validation and fallback:** The UI rejects IDs not in the filtered catalog, preserves the budget rule, and falls back to the deterministic local ranking if the API is unavailable or its output is malformed.

## Notes

- Prices are historical INR prices from the provided Flipkart sample, so product availability and links may no longer be current.
- The CSV is ignored by Git to avoid committing the 38 MB source data. To use another catalog, replace the file at the same path and retain the expected fields (or adjust `REQUIRED_COLUMNS` in `app.py`).
