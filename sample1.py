import pandas as pd
import duckdb
from faker import Faker
import random
from datetime import timedelta
from pathlib import Path

fake = Faker()
Faker.seed(42)
random.seed(42)

# -----------------------------------
# Channel configuration
# -----------------------------------
CHANNEL_CONFIG = {
    "Paid Search": {
        "base_traffic": 1.3,
        "conv_rate": 0.28,
        "value_min": 120,
        "value_max": 220,
        "cost_per_click": 2.5,  # 🆕 For RL: cost of each touch
    },
    "Organic Search": {
        "base_traffic": 1.0,
        "conv_rate": 0.20,
        "value_min": 80,
        "value_max": 180,
        "cost_per_click": 0.5,
    },
    "Social Media": {
        "base_traffic": 1.1,
        "conv_rate": 0.18,
        "value_min": 60,
        "value_max": 160,
        "cost_per_click": 1.8,
    },
    "Email": {
        "base_traffic": 0.9,
        "conv_rate": 0.25,
        "value_min": 70,
        "value_max": 190,
        "cost_per_click": 0.3,
    },
    "Direct": {
        "base_traffic": 0.8,
        "conv_rate": 0.22,
        "value_min": 90,
        "value_max": 200,
        "cost_per_click": 0.0,
    },
    "Unknown": {
        "base_traffic": 0.5,
        "conv_rate": 0.10,
        "value_min": 50,
        "value_max": 150,
        "cost_per_click": 0.0,
    },
}

CHANNELS = list(CHANNEL_CONFIG.keys())


def _compute_channel_probs(budget_weights=None):
    """
    Combine base_traffic with optional budget_weights to get
    final sampling probabilities per channel.
    """
    weights = {}
    for ch, cfg in CHANNEL_CONFIG.items():
        base = cfg["base_traffic"]
        if budget_weights and ch in budget_weights:
            weights[ch] = base * budget_weights[ch]
        else:
            weights[ch] = base

    total = sum(weights.values()) or 1.0
    probs = {ch: w / total for ch, w in weights.items()}
    return probs


def generate_messy_data(num_records=100000, budget_weights=None, enable_glitches=False):
    """
    Generate synthetic clickstream data.
    
    - num_records ≈ number of touchpoints (rows), not users.
    - budget_weights: optional dict to bias channel traffic.
    - enable_glitches: if True, 10% of users have backwards timestamps (default: False for clean data)
    """
    data = []
    channel_probs = _compute_channel_probs(budget_weights)

    # Precompute cumulative distribution for fast sampling
    cumulative = []
    running = 0.0
    for ch in CHANNELS:
        running += channel_probs[ch]
        cumulative.append((running, ch))

    def sample_channel():
        r = random.random()
        for cutoff, ch in cumulative:
            if r <= cutoff:
                return ch
        return CHANNELS[-1]

    approx_users = num_records // 3  # on average ~3 touches per user
    
    for _ in range(approx_users):
        user_id = fake.uuid4()

        # 🔧 FIXED: Only enable glitches if explicitly requested
        is_glitch = enable_glitches and (random.random() < 0.1)

        # 1–5 touchpoints per user
        num_touches = random.randint(1, 5)
        base_time = fake.date_time_between(start_date='-30d', end_date='now')

        user_events = []
        
        # 🆕 For RL: Track user engagement level
        user_engagement = random.uniform(0.3, 1.0)  # engagement score

        for i in range(num_touches):
            time_offset = timedelta(hours=random.randint(1, 48) * i)
            
            if is_glitch and i > 0:
                event_time = base_time - time_offset
            else:
                event_time = base_time + time_offset

            channel = sample_channel()
            
            # 🆕 Calculate cost for this touch
            cost = CHANNEL_CONFIG[channel]['cost_per_click']

            user_events.append({
                'event_id': fake.uuid4(),
                'user_id': user_id,
                'timestamp': event_time,
                'channel': channel,
                'conversion': 0,
                'conversion_value': 0.0,
                'cost': cost,  # 🆕 For RL
                'user_engagement': user_engagement,  # 🆕 For RL
                'touch_number': i + 1,  # 🆕 For RL: position in journey
            })

        # Decide conversion on the LAST touch
        last_channel = user_events[-1]['channel']
        conv_rate = CHANNEL_CONFIG[last_channel]['conv_rate']
        
        # 🆕 Engagement affects conversion rate
        adjusted_conv_rate = conv_rate * user_engagement

        did_convert = random.random() < adjusted_conv_rate

        if did_convert:
            vmin = CHANNEL_CONFIG[last_channel]['value_min']
            vmax = CHANNEL_CONFIG[last_channel]['value_max']
            conv_value = round(random.uniform(vmin, vmax), 2)

            user_events[-1]['conversion'] = 1
            user_events[-1]['conversion_value'] = conv_value

        data.extend(user_events)

    return pd.DataFrame(data)


if __name__ == "__main__":
    print("🌪️ Generating Clean Data for Attribution + RL...")

    df = generate_messy_data(num_records=100000, enable_glitches=False)

    # Save as CSV — upload this via the dashboard to run attribution
    csv_path = Path(__file__).resolve().parent / "generated_clicks.csv"
    df.to_csv(csv_path, index=False)

    print(f"✅ Saved {len(df):,} rows  →  {csv_path.name}")
    print("\n📊 Data Summary:")
    print(f"  Total touches: {len(df):,}")
    print(f"  Unique users:  {df['user_id'].nunique():,}")
    print(f"  Conversions:   {df['conversion'].sum():,}")
    print(f"  Total revenue: ${df['conversion_value'].sum():,.2f}")
    print(f"  Total cost:    ${df['cost'].sum():,.2f}")
    cost, rev = df['cost'].sum(), df['conversion_value'].sum()
    print(f"  ROI:           {((rev - cost) / cost * 100):.1f}%")
    print(f"\n👉 Upload '{csv_path.name}' via the dashboard sidebar to run attribution.")