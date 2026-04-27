"""
Scenarios (row distribution):
  normal                : 250 rows  — healthy, balanced resource
  over_provisioned      : 150 rows  — idle server, wasted cost
  cpu_saturated         : 200 rows  — compute maxed out
  memory_pressure       : 150 rows  — RAM near exhaustion
  traffic_cpu_mismatch  : 250 rows  — CPU and network moving in opposite directions
                                      (covers both runaway-process and exfil sub-patterns)
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

def clamp(value: float, low: float = 1.0, high: float = 100.0) -> float:
    """Keep metric values inside a realistic [low, high] range."""
    return float(max(low, min(high, value)))


def noisy(base: float, std: float) -> float:
    """Return base + Gaussian noise, clamped to [1, 100]."""
    return clamp(base + np.random.normal(0, std))

SCENARIOS = [
    {
        "label"               : "normal",
        "n"                   : 250,
        "cpu_avg_base"        : 45,       # Higher baseline
        "cpu_p95_base"        : 65,
        "memory_avg_base"     : 55,
        "network_pct_base"    : 50,
        "internet_facing_prob": 0.40,
        "identity_prob"       : 0.40,
        "noise_std"           : 15,       # Doubled noise
    },
    {
        # Very low CPU + low network → zombie/idle server, paying for nothing
        "label"               : "over_provisioned",
        "n"                   : 150,
        "cpu_avg_base"        : 12,       # Moved closer to normal
        "cpu_p95_base"        : 20,
        "memory_avg_base"     : 45,
        "network_pct_base"    : 18,
        "internet_facing_prob": 0.50,
        "identity_prob"       : 0.60,
        "noise_std"           : 12,       # Higher noise
    },
    {
        # CPU hammered, network proportionally high → heavy load / traffic surge
        "label"               : "cpu_saturated",
        "n"                   : 200,
        "cpu_avg_base"        : 80,       # Lowered anchor
        "cpu_p95_base"        : 90,
        "memory_avg_base"     : 65,
        "network_pct_base"    : 55,
        "internet_facing_prob": 0.55,
        "identity_prob"       : 0.45,
        "noise_std"           : 14,       # Higher noise
    },
    {
        # RAM near exhaustion while CPU is still moderate → memory leak or bloat
        "label"               : "memory_pressure",
        "n"                   : 150,
        "cpu_avg_base"        : 35,
        "cpu_p95_base"        : 50,
        "memory_avg_base"     : 85,       # Lowered anchor
        "network_pct_base"    : 35,
        "internet_facing_prob": 0.35,
        "identity_prob"       : 0.50,
        "noise_std"           : 14,
    },
    {
        # CPU and network move in OPPOSITE directions:
        #   sub-pattern A → high CPU, low network  = runaway process / infinite loop
        #   sub-pattern B → low CPU, high network  = data moving with no compute (exfil risk)
        # We split the 250 rows evenly between these two sub-patterns.
        "label"               : "traffic_cpu_mismatch",
        "n"                   : 250,          # handled specially below
        "cpu_avg_base"        : None,         # placeholder — see generation loop
        "cpu_p95_base"        : None,
        "memory_avg_base"     : None,
        "network_pct_base"    : None,
        "internet_facing_prob": 0.65,         # more likely exposed (higher risk)
        "identity_prob"       : 0.55,
        "noise_std"           : 15,
    },
]

def build_row(cpu_avg_base, cpu_p95_base, memory_avg_base,
              network_pct_base, internet_facing_prob, identity_prob,
              noise_std, label, row_id):
    """Generate one row of infrastructure metrics."""

    cpu_avg    = noisy(cpu_avg_base, noise_std)
    # p95 must always be >= avg; add a small positive offset then noise
    cpu_p95    = clamp(cpu_avg + noisy(cpu_p95_base - cpu_avg_base, noise_std * 0.5),
                       cpu_avg, 100)
    memory_avg = noisy(memory_avg_base, noise_std)
    network_pct = noisy(network_pct_base, noise_std)

    internet_facing  = bool(np.random.rand() < internet_facing_prob)
    identity_attached = bool(np.random.rand() < identity_prob)

    # ── engineered features ──────────────────────────────────────────────
    #
    # 1. cpu_spike_ratio  — how spiky is the CPU?
    #    p95 / avg  →  ratio ≈ 1 = stable load,  ratio > 2.5 = micro-bursting
    cpu_spike_ratio     = round(cpu_p95 / (cpu_avg + 1e-5), 3)

    # 2. resource_saturation — combined pressure on compute + memory
    #    (cpu_avg + memory_avg) / 2  →  > 70 signals a dual-pressure situation
    resource_saturation = round((cpu_avg + memory_avg) / 2, 3)

    # 3. load_efficiency — how much traffic per unit of CPU work?
    #    network_pct / (cpu_avg + 1)
    #    LOW value  = CPU working hard with little traffic  → runaway process
    #    HIGH value = lots of traffic with little CPU       → possible exfil
    load_efficiency     = round(network_pct / (cpu_avg + 1), 3)

    return {
        "resource_id"         : f"i-{row_id:04d}",
        "cpu_avg"             : round(cpu_avg, 1),
        "cpu_p95"             : round(cpu_p95, 1),
        "memory_avg"          : round(memory_avg, 1),
        "network_pct"         : round(network_pct, 1),
        "internet_facing"     : internet_facing,
        "identity_attached"   : identity_attached,
        # engineered
        "cpu_spike_ratio"     : cpu_spike_ratio,
        "resource_saturation" : resource_saturation,
        "load_efficiency"     : load_efficiency,
        # labels
        "anomaly_type"        : label,
        "is_anomalous"        : label != "normal",
    }


def generate_dataset() -> pd.DataFrame:
    rows   = []
    row_id = 1

    for scenario in SCENARIOS:
        label  = scenario["label"]
        n      = scenario["n"]
        std    = scenario["noise_std"]
        if_prob = scenario["internet_facing_prob"]
        id_prob = scenario["identity_prob"]

        if label == "traffic_cpu_mismatch":
            # Sub-pattern A: high CPU, low network (runaway process)
            half = n // 2
            for _ in range(half):
                rows.append(build_row(
                    cpu_avg_base=75, cpu_p95_base=85, # Less extreme
                    memory_avg_base=50, network_pct_base=25,
                    internet_facing_prob=if_prob, identity_prob=id_prob,
                    noise_std=std, label=label, row_id=row_id
                ))
                row_id += 1

            # Sub-pattern B: low CPU, high network (data exfiltration signal)
            for _ in range(n - half):
                rows.append(build_row(
                    cpu_avg_base=15, cpu_p95_base=25,
                    memory_avg_base=40, network_pct_base=75,
                    internet_facing_prob=if_prob, identity_prob=id_prob,
                    noise_std=std, label=label, row_id=row_id
                ))
                row_id += 1
        else:
            for _ in range(n):
                rows.append(build_row(
                    cpu_avg_base=scenario["cpu_avg_base"],
                    cpu_p95_base=scenario["cpu_p95_base"],
                    memory_avg_base=scenario["memory_avg_base"],
                    network_pct_base=scenario["network_pct_base"],
                    internet_facing_prob=if_prob,
                    identity_prob=id_prob,
                    noise_std=std,
                    label=label,
                    row_id=row_id,
                ))
                row_id += 1

    df = pd.DataFrame(rows)

    # --- NEW: Introduce Label Corruption (Realistic Errors) ---
    # Randomly flip 5% of labels to a different class
    corrupt_idx = df.sample(frac=0.05).index
    possible_labels = [s["label"] for s in SCENARIOS]
    for idx in corrupt_idx:
        current_label = df.loc[idx, "anomaly_type"]
        new_label = np.random.choice([l for l in possible_labels if l != current_label])
        df.loc[idx, "anomaly_type"] = new_label
        df.loc[idx, "is_anomalous"] = (new_label != "normal")

    # shuffle so the CSV isn't grouped by scenario
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = generate_dataset()

    # ── save ──────────────────────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "synthetic_metrics.csv")
    df.to_csv(out_path, index=False)

    # ── quick sanity check ────────────────────────────────────────────────────
    print(f"Dataset saved to: {os.path.abspath(out_path)}")
    print(f"Total rows       : {len(df)}")
    print(f"\nClass distribution:")
    print(df["anomaly_type"].value_counts().to_string())
    print(f"\nAnomaly split    : {df['is_anomalous'].sum()} anomalous / "
          f"{(~df['is_anomalous']).sum()} normal")
    print(f"\nSample rows:")
    print(df.head(5).to_string(index=False))
