import unittest

import pandas as pd

import app


def sample_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pid": "shoe-1",
                "catalog_id": "one",
                "product_name": "Comfort Runner Shoe",
                "brand": "Aster",
                "category": "Footwear",
                "category_path": "Footwear >> Men's Footwear >> Sports Shoes",
                "description_short": "Comfortable casual daily running shoe with foam sole",
                "price": 1200.0,
                "retail_price": 1800.0,
                "rating": 4.4,
                "discount_percent": 33.3,
                "image_url": "",
                "product_url": "https://example.test/shoe-1",
                "search_text": "comfort runner shoe aster footwear comfortable casual daily running shoe with foam sole",
            },
            {
                "pid": "shoe-2",
                "catalog_id": "two",
                "product_name": "Formal Leather Shoe",
                "brand": "Aster",
                "category": "Footwear",
                "category_path": "Footwear >> Men's Footwear >> Formal Shoes",
                "description_short": "Leather office shoe",
                "price": 3200.0,
                "retail_price": 3200.0,
                "rating": 4.8,
                "discount_percent": 0.0,
                "image_url": "",
                "product_url": "https://example.test/shoe-2",
                "search_text": "formal leather shoe aster footwear leather office shoe",
            },
            {
                "pid": "bag-1",
                "catalog_id": "three",
                "product_name": "Casual Backpack",
                "brand": "Bags Co",
                "category": "Bags",
                "category_path": "Bags >> Backpacks",
                "description_short": "Daily backpack",
                "price": 900.0,
                "retail_price": 1200.0,
                "rating": 4.2,
                "discount_percent": 25.0,
                "image_url": "",
                "product_url": "https://example.test/bag-1",
                "search_text": "casual backpack bags co bags daily backpack",
            },
        ]
    )


class CatalogHelpersTests(unittest.TestCase):
    def test_filtering_keeps_category_and_budget_as_hard_rules(self):
        results = app.search_products(
            sample_catalog(), "Footwear", 2_000, "comfortable casual running shoe", "Best match"
        )
        self.assertEqual(results["catalog_id"].tolist(), ["one"])
        self.assertTrue((results["price"] <= 2_000).all())
        self.assertTrue((results["category"] == "Footwear").all())

    def test_candidate_payload_is_compact_and_json_serializable(self):
        payload = app.candidate_payload(sample_catalog())
        self.assertEqual(payload[0]["catalog_id"], "one")
        self.assertEqual(payload[0]["price"], 1200.0)
        self.assertNotIn("product_url", payload[0])

    def test_json_extraction_allows_markdown_fence(self):
        data = app.extract_json('```json\n{"recommendations": []}\n```')
        self.assertEqual(data, {"recommendations": []})

    def test_local_fallback_returns_at_most_three_ranked_items(self):
        recommendations, _ = app.local_recommendations(sample_catalog(), "daily use")
        self.assertEqual(len(recommendations), 3)
        self.assertEqual(recommendations[0]["catalog_id"], "one")


if __name__ == "__main__":
    unittest.main()
