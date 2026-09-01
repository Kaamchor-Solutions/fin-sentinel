"""Central configuration for FinSentinel.

All domain constants live here so that the data generator, the feature
pipeline and the serving layer always agree on vocabulary and defaults.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Domain constants
# --------------------------------------------------------------------------

#: Merchant categories used by the synthetic transaction generator.
CATEGORIES = (
    "groceries",
    "fuel",
    "dining",
    "fashion",
    "healthcare",
    "subscriptions",
    "digital_goods",
    "electronics",
    "travel",
    "cash_equivalent",
)

#: Static, analyst-supplied risk weight per category (0 = safe, 1 = risky).
#: This encodes domain knowledge (e.g. cash equivalents and digital goods are
#: favoured by fraudsters because they are easy to resell) and is injected as
#: a feature rather than learned from the labels.
CATEGORY_RISK = {
    "groceries": 0.10,
    "fuel": 0.15,
    "dining": 0.20,
    "fashion": 0.25,
    "healthcare": 0.20,
    "subscriptions": 0.30,
    "travel": 0.40,
    "electronics": 0.50,
    "digital_goods": 0.80,
    "cash_equivalent": 0.90,
}

#: The exact feature vector the models are trained on. The scoring API
#: recomputes these from raw request fields, so clients never see them.
MODEL_FEATURES = [
    "amount",
    "log_amount",
    "hour_sin",
    "hour_cos",
    "is_night",
    "is_weekend",
    "amount_vs_user_mean",
    "minutes_since_prior_txn",
    "txn_count_1h",
    "txn_count_24h",
    "category_risk",
]

#: Risk bands used by the scoring API to translate a probability into an
#: operational decision. Illustrative defaults — tune them for your portfolio.
RISK_BANDS = (
    (0.0, 0.20, "low"),
    (0.20, 0.60, "medium"),
    (0.60, 1.01, "high"),
)

#: Actions the API recommends per risk band.
RECOMMENDATIONS = {
    "low": "approve",
    "medium": "step-up authentication (OTP / 3-D Secure)",
    "high": "block and route to manual review",
}

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

DEFAULT_ARTIFACT_DIR = Path("artifacts")
DEFAULT_DATA_DIR = Path("data")
