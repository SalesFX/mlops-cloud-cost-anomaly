#!/usr/bin/env python3
"""Synthetic cloud cost dataset generator for the FinOps anomaly detection platform.

Usage:
    python src/ml/generate_dataset.py --days 180 --output data/cloud_costs.csv --seed 42 --anomaly-rate 0.05
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "date",
    "provider",
    "account_id",
    "service",
    "region",
    "environment",
    "resource_id",
    "tag_project",
    "tag_owner",
    "daily_cost",
    "usage_quantity",
    "currency",
    "is_anomaly",
    "anomaly_type",
]

# ---------------------------------------------------------------------------
# Cloud provider catalog
# ---------------------------------------------------------------------------

PROVIDERS = {
    "AWS": {
        "services": ["EC2", "RDS", "S3", "Lambda", "EKS"],
        "regions": ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
        "account_ids": ["123456789012", "987654321098", "112233445566"],
    },
    "OCI": {
        "services": [
            "Compute",
            "Autonomous Database",
            "Object Storage",
            "OKE",
            "Load Balancer",
        ],
        "regions": ["us-ashburn-1", "us-phoenix-1", "eu-frankfurt-1", "ap-tokyo-1"],
        "account_ids": [
            "ocid1.tenancy.oc1..aaa123456789",
            "ocid1.tenancy.oc1..bbb987654321",
        ],
    },
}

ENVIRONMENTS = ["dev", "staging", "prod"]

# prod costs ~4.5x dev; staging ~1.8x dev
ENV_COST_MULTIPLIER = {"dev": 1.0, "staging": 1.8, "prod": 4.5}

# ---------------------------------------------------------------------------
# Service cost and usage baselines
# ---------------------------------------------------------------------------

SERVICE_BASE_COST = {
    "EC2": 50.0,
    "RDS": 80.0,
    "S3": 5.0,
    "Lambda": 2.0,
    "EKS": 100.0,
    "Compute": 50.0,
    "Autonomous Database": 100.0,
    "Object Storage": 3.0,
    "OKE": 80.0,
    "Load Balancer": 10.0,
}

SERVICE_BASE_USAGE = {
    "EC2": 720.0,
    "RDS": 720.0,
    "S3": 1000.0,
    "Lambda": 100000.0,
    "EKS": 720.0,
    "Compute": 720.0,
    "Autonomous Database": 720.0,
    "Object Storage": 1000.0,
    "OKE": 720.0,
    "Load Balancer": 720.0,
}

# ---------------------------------------------------------------------------
# Anomaly and dimension catalogs
# ---------------------------------------------------------------------------

ANOMALY_TYPES_LIST = [
    "cost_spike",
    "usage_spike",
    "missing_tag",
    "unexpected_service_growth",
]

PROJECTS = ["project-alpha", "project-beta", "project-gamma", "project-delta"]
OWNERS = ["alice", "bob", "charlie", "diana"]

RESOURCES_PER_COMBO = 2  # synthetic resource instances per (provider, service, env)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_RESOURCE_ID_TEMPLATES = {
    "EC2": "i-{h}",
    "RDS": "db-{h}",
    "S3": "s3-{h}",
    "Lambda": "fn-{h}",
    "EKS": "eks-{h}",
    "Compute": "ocid1.instance.oc1.{h}",
    "Autonomous Database": "ocid1.adb.oc1.{h}",
    "Object Storage": "ocid1.bucket.oc1.{h}",
    "OKE": "ocid1.cluster.oc1.{h}",
    "Load Balancer": "ocid1.lb.oc1.{h}",
}


def _hex(rng: np.random.Generator, size: int = 8) -> str:
    return "".join(rng.choice(list("0123456789abcdef"), size=size).tolist())


def _build_resource_catalog(rng: np.random.Generator) -> dict:
    """Pre-generate stable resource IDs so each resource appears consistently across days."""
    catalog = {}
    for provider, pconfig in PROVIDERS.items():
        for service in pconfig["services"]:
            template = _RESOURCE_ID_TEMPLATES.get(service, "resource-{h}")
            for env in ENVIRONMENTS:
                for r in range(RESOURCES_PER_COMBO):
                    catalog[(provider, service, env, r)] = template.format(h=_hex(rng))
    return catalog


# ---------------------------------------------------------------------------
# Core generation functions
# ---------------------------------------------------------------------------

def generate_normal_records(
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
    resource_catalog: dict,
) -> list:
    records = []
    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        for provider, pconfig in PROVIDERS.items():
            for service in pconfig["services"]:
                base_cost = SERVICE_BASE_COST[service]
                base_usage = SERVICE_BASE_USAGE[service]
                for env in ENVIRONMENTS:
                    env_mult = ENV_COST_MULTIPLIER[env]
                    region = rng.choice(pconfig["regions"])
                    account_id = rng.choice(pconfig["account_ids"])
                    for r in range(RESOURCES_PER_COMBO):
                        cost = rng.lognormal(
                            mean=np.log(base_cost * env_mult),
                            sigma=0.25,
                        )
                        usage = rng.lognormal(
                            mean=np.log(base_usage * env_mult),
                            sigma=0.25,
                        )
                        records.append(
                            {
                                "date": date_str,
                                "provider": provider,
                                "account_id": account_id,
                                "service": service,
                                "region": region,
                                "environment": env,
                                "resource_id": resource_catalog[(provider, service, env, r)],
                                "tag_project": rng.choice(PROJECTS),
                                "tag_owner": rng.choice(OWNERS),
                                "daily_cost": round(float(cost), 4),
                                "usage_quantity": round(float(usage), 2),
                                "currency": "USD",
                                "is_anomaly": False,
                                "anomaly_type": "none",
                            }
                        )
    return records


def inject_anomalies(
    df: pd.DataFrame,
    anomaly_rate: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n_anomalies = min(int(len(df) * anomaly_rate), len(df))
    if n_anomalies == 0:
        return df

    indices = rng.choice(df.index.to_numpy(), size=n_anomalies, replace=False)

    for idx in indices:
        atype = rng.choice(ANOMALY_TYPES_LIST)
        df.at[idx, "is_anomaly"] = True
        df.at[idx, "anomaly_type"] = atype

        if atype == "cost_spike":
            mult = float(rng.uniform(5.0, 20.0))
            df.at[idx, "daily_cost"] = round(float(df.at[idx, "daily_cost"]) * mult, 4)

        elif atype == "usage_spike":
            mult = float(rng.uniform(5.0, 20.0))
            df.at[idx, "usage_quantity"] = round(float(df.at[idx, "usage_quantity"]) * mult, 2)
            df.at[idx, "daily_cost"] = round(float(df.at[idx, "daily_cost"]) * mult * 0.9, 4)

        elif atype == "missing_tag":
            df.at[idx, "tag_project"] = ""
            df.at[idx, "tag_owner"] = ""

        elif atype == "unexpected_service_growth":
            mult = float(rng.uniform(2.0, 5.0))
            df.at[idx, "daily_cost"] = round(float(df.at[idx, "daily_cost"]) * mult, 4)
            df.at[idx, "usage_quantity"] = round(float(df.at[idx, "usage_quantity"]) * mult, 2)

    return df


def generate_dataset(
    days: int,
    seed: int,
    anomaly_rate: float,
    end_date: str = None,
) -> pd.DataFrame:
    if not 0.0 <= anomaly_rate <= 1.0:
        raise ValueError(f"anomaly_rate must be between 0.0 and 1.0, got {anomaly_rate}")

    rng = np.random.default_rng(seed)

    end = pd.Timestamp(end_date) if end_date else pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=days - 1)
    dates = pd.date_range(start=start, end=end, freq="D")

    resource_catalog = _build_resource_catalog(rng)
    records = generate_normal_records(dates, rng, resource_catalog)
    df = pd.DataFrame(records)
    df = inject_anomalies(df, anomaly_rate, rng)
    df = df.sort_values(["date", "provider", "service", "environment"]).reset_index(drop=True)
    return df


def save_dataset(df: pd.DataFrame, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} records to {path}")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic cloud cost dataset with controlled anomaly injection."
    )
    parser.add_argument("--days", type=int, default=180, help="Number of days (default: 180)")
    parser.add_argument(
        "--output", type=str, default="data/cloud_costs.csv", help="Output CSV path"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--anomaly-rate",
        type=float,
        default=0.05,
        help="Fraction of anomalous records (default: 0.05)",
    )
    args = parser.parse_args()

    df = generate_dataset(days=args.days, seed=args.seed, anomaly_rate=args.anomaly_rate)
    save_dataset(df, args.output)

    print(f"\nDataset summary:")
    print(f"  Total records  : {len(df)}")
    print(f"  Anomaly rate   : {df['is_anomaly'].mean():.2%}")
    print(f"  Date range     : {df['date'].min()} to {df['date'].max()}")


if __name__ == "__main__":
    main()
