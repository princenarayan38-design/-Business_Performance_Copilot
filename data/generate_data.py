"""
generate_data.py

Generates a realistic synthetic business dataset for BusinessIntelligence.ai.

WHAT THIS SCRIPT DOES
----------------------
Creates four CSV files (sales, customer_feedback, delivery, events) that together
describe ~26 weeks of daily business activity across 4 regions and 4 products.

A single, deliberately-designed business event is embedded: a logistics
disruption in the NORTH region during weeks 15-18, which causes:
    1. average_delivery_days to rise (delivery.csv)
    2. late_delivery_rate to rise (delivery.csv)
    3. negative, delivery-related customer feedback to spike (customer_feedback.csv)
    4. units_sold in North to drop, with a short lag (sales.csv)

IMPORTANT: this disruption constant (region="North", weeks 15-18) is used ONLY
in this file, to build the data. No downstream module (anomaly_detector.py,
hypothesis_generator.py, etc.) will ever read DISRUPTION_REGION or
DISRUPTION_WEEKS - they must discover the pattern from the numbers alone.

Background noise is added so the disruption isn't the only thing that moves:
    - random day-to-day sales fluctuation in every region
    - an unrelated price increase on one product in South (a red herring -
      it does NOT cause a sales drop, to test that the system doesn't
      wrongly blame every price change)
    - a scattering of baseline-rate negative feedback in all regions
    - two legitimate promotions (recorded in events.csv) that cause real,
      explainable sales bumps

Run:
    python data/generate_data.py

Output:
    data/raw/sales.csv
    data/raw/customer_feedback.csv
    data/raw/delivery.csv
    data/raw/events.csv
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
START_DATE = date(2026, 1, 5)  # a Monday
NUM_WEEKS = 26
NUM_DAYS = NUM_WEEKS * 7

REGIONS = ["North", "South", "East", "West"]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "raw")

# The embedded event. Used ONLY in this generator file.
DISRUPTION_REGION = "North"
DISRUPTION_START_WEEK = 15
DISRUPTION_END_WEEK = 18  # inclusive
DISRUPTION_START_DAY = (DISRUPTION_START_WEEK - 1) * 7
DISRUPTION_END_DAY = DISRUPTION_END_WEEK * 7  # exclusive upper bound

# A red-herring price change: real, but with no causal effect on sales.
NOISE_PRICE_HIKE_REGION = "South"
NOISE_PRICE_HIKE_START_WEEK = 10
NOISE_PRICE_HIKE_PRODUCT = "P002"

rng = np.random.default_rng(RANDOM_SEED)


@dataclass
class Product:
    product_id: str
    product_name: str
    base_price: float
    base_daily_units_per_region: float


PRODUCTS = [
    Product("P001", "AirWave Earbuds Pro", base_price=2499.0, base_daily_units_per_region=18),
    Product("P002", "PulseFit Smart Watch", base_price=4999.0, base_daily_units_per_region=10),
    Product("P003", "EchoBeam Speaker Mini", base_price=1799.0, base_daily_units_per_region=14),
    Product("P004", "ShieldCase Armor", base_price=599.0, base_daily_units_per_region=25),
]


def date_for_day_index(day_index: int) -> date:
    return START_DATE + timedelta(days=day_index)


def week_of_day_index(day_index: int) -> int:
    """1-indexed week number."""
    return day_index // 7 + 1


# ---------------------------------------------------------------------------
# events.csv
# ---------------------------------------------------------------------------

def generate_events() -> pd.DataFrame:
    rows = []

    rows.append({
        "date": date_for_day_index(DISRUPTION_START_DAY),
        "event_type": "operational_disruption",
        "description": (
            f"Third-party logistics partner reported warehouse capacity issues "
            f"affecting the {DISRUPTION_REGION} distribution hub."
        ),
    })
    rows.append({
        "date": date_for_day_index(DISRUPTION_END_DAY - 1),
        "event_type": "operational_recovery",
        "description": f"{DISRUPTION_REGION} distribution hub logistics issue resolved.",
    })

    # Two legitimate, explainable promotions elsewhere - real signal, not noise.
    rows.append({
        "date": date_for_day_index(6 * 7),
        "event_type": "promotion_start",
        "description": "East region: 15% promotional discount on EchoBeam Speaker Mini (1 week).",
    })
    rows.append({
        "date": date_for_day_index(6 * 7 + 6),
        "event_type": "promotion_end",
        "description": "East region: EchoBeam Speaker Mini promotion ended.",
    })
    rows.append({
        "date": date_for_day_index(21 * 7),
        "event_type": "promotion_start",
        "description": "West region: 10% promotional discount on AirWave Earbuds Pro (1 week).",
    })
    rows.append({
        "date": date_for_day_index(21 * 7 + 6),
        "event_type": "promotion_end",
        "description": "West region: AirWave Earbuds Pro promotion ended.",
    })

    # The unrelated (red-herring) price hike.
    rows.append({
        "date": date_for_day_index((NOISE_PRICE_HIKE_START_WEEK - 1) * 7),
        "event_type": "price_change",
        "description": (
            f"{NOISE_PRICE_HIKE_REGION} region: price increased for product "
            f"{NOISE_PRICE_HIKE_PRODUCT} (routine supplier cost adjustment)."
        ),
    })

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# delivery.csv
# ---------------------------------------------------------------------------

def generate_delivery() -> pd.DataFrame:
    rows = []
    for day_index in range(NUM_DAYS):
        d = date_for_day_index(day_index)
        for region in REGIONS:
            for product in PRODUCTS:
                base_days = 3.0 + rng.normal(0, 0.3)
                base_late_rate = 0.04 + rng.normal(0, 0.01)

                if (
                    region == DISRUPTION_REGION
                    and DISRUPTION_START_DAY <= day_index < DISRUPTION_END_DAY
                ):
                    # Ramp up, hold, ramp down rather than a hard step -
                    # more realistic and still clearly detectable.
                    days_into = day_index - DISRUPTION_START_DAY
                    ramp = min(1.0, (days_into + 1) / 4.0)
                    base_days += ramp * 3.5 + rng.normal(0, 0.2)
                    base_late_rate += ramp * 0.35 + rng.normal(0, 0.02)

                rows.append({
                    "date": d,
                    "region": region,
                    "product_id": product.product_id,
                    "average_delivery_days": round(max(base_days, 0.5), 2),
                    "late_delivery_rate": round(min(max(base_late_rate, 0.0), 1.0), 3),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# sales.csv
# ---------------------------------------------------------------------------

def generate_sales(events_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    promo_windows = {}  # (region, product_id) -> set of day_index with discount
    for _, ev in events_df[events_df.event_type == "promotion_start"].iterrows():
        region = "East" if "East" in ev.description else "West"
        product = next(p for p in PRODUCTS if p.product_name in ev.description)
        start_day = (ev.date - START_DATE).days
        promo_windows[(region, product.product_id)] = set(range(start_day, start_day + 7))

    for day_index in range(NUM_DAYS):
        d = date_for_day_index(day_index)
        week = week_of_day_index(day_index)

        for region in REGIONS:
            for product in PRODUCTS:
                base_units = product.base_daily_units_per_region
                base_units *= rng.normal(1.0, 0.08)  # normal daily noise, every region

                price = product.base_price
                discount = 0.0

                # Red-herring price hike (South, P002, from week 10 onward): real
                # price change, no sales effect - a hypothesis the system should
                # find weak evidence for.
                if (
                    region == NOISE_PRICE_HIKE_REGION
                    and product.product_id == NOISE_PRICE_HIKE_PRODUCT
                    and week >= NOISE_PRICE_HIKE_START_WEEK
                ):
                    price *= 1.12

                # Legitimate promotions -> real, explainable sales bump.
                if (region, product.product_id) in promo_windows and day_index in promo_windows[(region, product.product_id)]:
                    discount = 0.15 if region == "East" else 0.10
                    base_units *= 1.6

                # The disruption's effect on sales, lagged ~2 days behind the
                # delivery/complaint signal, and easing as the fix takes hold.
                if region == DISRUPTION_REGION:
                    lag = 2
                    effective_day = day_index - lag
                    if DISRUPTION_START_DAY <= effective_day < DISRUPTION_END_DAY + 5:
                        days_into = effective_day - DISRUPTION_START_DAY
                        ramp = min(1.0, max(0.0, (days_into + 1) / 5.0))
                        # tapers off over the 5 days after the fix
                        taper = 1.0
                        if effective_day >= DISRUPTION_END_DAY:
                            taper = max(0.0, 1 - (effective_day - DISRUPTION_END_DAY) / 5.0)
                        base_units *= (1 - 0.30 * ramp * max(ramp, taper) if effective_day < DISRUPTION_END_DAY else 1 - 0.30 * taper)

                marketing_spend = round(max(rng.normal(150, 30), 0), 2)
                units_sold = max(int(round(base_units)), 0)
                revenue = round(units_sold * price * (1 - discount), 2)
                inventory = int(rng.integers(50, 400))

                rows.append({
                    "date": d,
                    "product_id": product.product_id,
                    "product_name": product.product_name,
                    "region": region,
                    "units_sold": units_sold,
                    "price": round(price, 2),
                    "revenue": revenue,
                    "marketing_spend": marketing_spend,
                    "discount": discount,
                    "inventory": inventory,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# customer_feedback.csv
# ---------------------------------------------------------------------------

POSITIVE_SNIPPETS = [
    "Really happy with the product quality.",
    "Works exactly as described, would buy again.",
    "Great value for the price.",
    "Fast setup and easy to use.",
]
NEUTRAL_SNIPPETS = [
    "It's okay, does what it says.",
    "Average experience, nothing special.",
    "Packaging could be better.",
]
NEGATIVE_GENERIC_SNIPPETS = [
    "Product stopped working after a week.",
    "Not as described on the listing.",
    "Customer support was slow to respond.",
]
NEGATIVE_DELIVERY_SNIPPETS = [
    "Order took way longer than expected to arrive.",
    "Delivery was delayed by several days with no update.",
    "Package arrived very late, missed the occasion it was for.",
    "Tracking showed no movement for days, delivery was delayed.",
]

CATEGORIES_NON_DELIVERY = ["product_quality", "pricing", "customer_support", "packaging"]


def generate_feedback() -> pd.DataFrame:
    rows = []
    customer_counter = 1

    for day_index in range(NUM_DAYS):
        d = date_for_day_index(day_index)
        for region in REGIONS:
            # Baseline feedback volume, every region every day.
            n_feedback = rng.poisson(1.5)

            is_disruption_window = (
                region == DISRUPTION_REGION
                and DISRUPTION_START_DAY <= day_index < DISRUPTION_END_DAY + 4
            )
            if is_disruption_window:
                n_feedback += rng.poisson(4)  # extra complaint volume

            for _ in range(n_feedback):
                product = PRODUCTS[rng.integers(0, len(PRODUCTS))]
                customer_id = f"C{customer_counter:05d}"
                customer_counter += 1

                if is_disruption_window and rng.random() < 0.75:
                    sentiment = "negative"
                    category = "delivery"
                    text = NEGATIVE_DELIVERY_SNIPPETS[rng.integers(0, len(NEGATIVE_DELIVERY_SNIPPETS))]
                else:
                    roll = rng.random()
                    if roll < 0.55:
                        sentiment = "positive"
                        category = "product_quality"
                        text = POSITIVE_SNIPPETS[rng.integers(0, len(POSITIVE_SNIPPETS))]
                    elif roll < 0.80:
                        sentiment = "neutral"
                        category = CATEGORIES_NON_DELIVERY[rng.integers(0, len(CATEGORIES_NON_DELIVERY))]
                        text = NEUTRAL_SNIPPETS[rng.integers(0, len(NEUTRAL_SNIPPETS))]
                    else:
                        sentiment = "negative"
                        category = CATEGORIES_NON_DELIVERY[rng.integers(0, len(CATEGORIES_NON_DELIVERY))]
                        text = NEGATIVE_GENERIC_SNIPPETS[rng.integers(0, len(NEGATIVE_GENERIC_SNIPPETS))]

                rows.append({
                    "date": d,
                    "customer_id": customer_id,
                    "region": region,
                    "product_id": product.product_id,
                    "feedback": text,
                    "sentiment": sentiment,
                    "category": category,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    events_df = generate_events()
    delivery_df = generate_delivery()
    sales_df = generate_sales(events_df)
    feedback_df = generate_feedback()

    events_df.to_csv(os.path.join(OUTPUT_DIR, "events.csv"), index=False)
    delivery_df.to_csv(os.path.join(OUTPUT_DIR, "delivery.csv"), index=False)
    sales_df.to_csv(os.path.join(OUTPUT_DIR, "sales.csv"), index=False)
    feedback_df.to_csv(os.path.join(OUTPUT_DIR, "customer_feedback.csv"), index=False)

    print(f"Wrote {len(sales_df)} sales rows")
    print(f"Wrote {len(feedback_df)} feedback rows")
    print(f"Wrote {len(delivery_df)} delivery rows")
    print(f"Wrote {len(events_df)} event rows")
    print(f"Files written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
