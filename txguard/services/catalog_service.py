from typing import Any, Dict, List, Optional

CHANNEL3_PRODUCTS: List[Dict[str, Any]] = [
    {
        "product_id": "CH3-SONY-XM5",
        "sku": "SNY-WH1000XM5-BLK",
        "title": "Sony WH-1000XM5 Headphones",
        "brand": "Sony",
        "category": "Consumer Electronics",
        "price": 348.00,
        "currency": "USD",
        "is_digital_goods": False,
        "merchant_name": "SoundWave Flagship Audio",
        "merchant_id": "MERCH-CH3-8821",
        "channel3_trust_rating": 0.98,
        "recommended_scenario": "LOW_RISK_LEGIT",
        "description": "Wireless noise-canceling headphones.",
        "preset_telemetry": {
            "payer_email": "alex.morgan@verified-buyer.com",
            "payer_name": "Alex Morgan",
            "payer_country": "US",
            "shipping_country": "US",
            "velocity_1h": 1,
            "account_age_days": 840,
            "country_mismatch": False,
            "ip_proxy_detected": False,
            "device_trust_score": 0.96,
            "is_digital_goods": False,
            "email_domain_risk": 0.02,
        },
    },
    {
        "product_id": "CH3-APPLE-IPAD-M4",
        "sku": "APL-IPADPRO13-512",
        "title": "Apple iPad Pro 13-inch",
        "brand": "Apple",
        "category": "Tablets & Computing",
        "price": 1299.00,
        "currency": "USD",
        "is_digital_goods": False,
        "merchant_name": "Apex Tech Authorized Reseller",
        "merchant_id": "MERCH-CH3-4109",
        "channel3_trust_rating": 0.94,
        "recommended_scenario": "MEDIUM_RISK_STEP_UP",
        "description": "13-inch tablet with M4 chip and 512GB storage.",
        "preset_telemetry": {
            "payer_email": "jordan.lee.travel@fastmail.io",
            "payer_name": "Jordan Lee",
            "payer_country": "US",
            "shipping_country": "CA",
            "velocity_1h": 3,
            "account_age_days": 28,
            "country_mismatch": True,
            "ip_proxy_detected": False,
            "device_trust_score": 0.58,
            "is_digital_goods": False,
            "email_domain_risk": 0.25,
        },
    },
    {
        "product_id": "CH3-NVIDIA-RTX5090",
        "sku": "NV-RTX5090-FE-32G",
        "title": "NVIDIA GeForce RTX 5090",
        "brand": "NVIDIA",
        "category": "GPUs & AI Accelerators",
        "price": 2499.00,
        "currency": "USD",
        "is_digital_goods": False,
        "merchant_name": "SiliconCore Compute Direct",
        "merchant_id": "MERCH-CH3-9902",
        "channel3_trust_rating": 0.91,
        "recommended_scenario": "HIGH_RISK_FRAUD",
        "description": "Founders Edition 32GB GDDR7 graphics card.",
        "preset_telemetry": {
            "payer_email": "temp.drop.9941@anon-mail.xyz",
            "payer_name": "V. Kova",
            "payer_country": "US",
            "shipping_country": "RO",
            "velocity_1h": 7,
            "account_age_days": 2,
            "country_mismatch": True,
            "ip_proxy_detected": True,
            "device_trust_score": 0.14,
            "is_digital_goods": False,
            "email_domain_risk": 0.91,
        },
    },
    {
        "product_id": "CH3-STEAM-VOUCHER-500",
        "sku": "VALVE-GC-500-INSTANT",
        "title": "Digital Gaming & Cloud Compute Gift Bundle",
        "brand": "Multi-Platform Digital",
        "category": "Digital Gift Cards",
        "price": 500.00,
        "currency": "USD",
        "is_digital_goods": True,
        "merchant_name": "InstantKey Digital Vault",
        "merchant_id": "MERCH-CH3-7740",
        "channel3_trust_rating": 0.86,
        "recommended_scenario": "HIGH_RISK_FRAUD",
        "description": "Digital gift card bundle.",
        "preset_telemetry": {
            "payer_email": "cardtest.bot88@disposable.org",
            "payer_name": "M. Noname",
            "payer_country": "GB",
            "shipping_country": "NG",
            "velocity_1h": 9,
            "account_age_days": 1,
            "country_mismatch": True,
            "ip_proxy_detected": True,
            "device_trust_score": 0.11,
            "is_digital_goods": True,
            "email_domain_risk": 0.95,
        },
    },
    {
        "product_id": "CH3-GARMIN-FENIX8",
        "sku": "GRM-FENIX8-47-SOLAR",
        "title": "Garmin Fenix 8 Sapphire Solar",
        "brand": "Garmin",
        "category": "Wearables & Fitness",
        "price": 1099.00,
        "currency": "USD",
        "is_digital_goods": False,
        "merchant_name": "TrailPeak Endurance Gear",
        "merchant_id": "MERCH-CH3-3315",
        "channel3_trust_rating": 0.97,
        "recommended_scenario": "LOW_RISK_LEGIT",
        "description": "Multisport GPS smartwatch with solar charging.",
        "preset_telemetry": {
            "payer_email": "elena.rostova@alumni.edu",
            "payer_name": "Elena Rostova",
            "payer_country": "US",
            "shipping_country": "US",
            "velocity_1h": 1,
            "account_age_days": 1420,
            "country_mismatch": False,
            "ip_proxy_detected": False,
            "device_trust_score": 0.95,
            "is_digital_goods": False,
            "email_domain_risk": 0.01,
        },
    },
]


def get_catalog() -> List[Dict[str, Any]]:
    """Returns the full Channel3 product intelligence catalog."""
    return CHANNEL3_PRODUCTS


def get_product_by_id(product_id: str) -> Optional[Dict[str, Any]]:
    """Finds a Channel3 catalog item by product_id."""
    for prod in CHANNEL3_PRODUCTS:
        if prod["product_id"].lower() == product_id.lower():
            return prod
    return None


def search_products(
    query: str = "",
    max_price: Optional[float] = None,
    category: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Searches the Channel3 product feed by keyword, max price, and category for Agentic Commerce."""
    q = (query or "").strip().lower()
    results = []
    for prod in CHANNEL3_PRODUCTS:
        if q:
            haystack = (
                f"{prod['product_id']} {prod['title']} {prod['brand']} "
                f"{prod['category']} {prod['description']}"
            ).lower()
            if q not in haystack:
                continue
        if max_price is not None and prod["price"] > max_price:
            continue
        if category and category.lower() not in prod["category"].lower():
            continue
        results.append(prod)
    return results

