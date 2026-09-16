from __future__ import annotations

import argparse
from pathlib import Path

from diffpred.data.polaris import PolarisDatamodule


def parse_csv_arg(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare cached Polaris benchmark split files for multitask runs."
    )
    parser.add_argument(
        "--datasets",
        default="biogen/adme-fang-reg-v1,graphium/tox21-v1,graphium/zinc12k-v1",
    )
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument(
        "--stamp-path",
        default="artifacts/data/multitask/polaris_benchmark_splits_prepared.txt",
    )
    args = parser.parse_args()

    datasets = parse_csv_arg(args.datasets)
    seeds = [int(seed) for seed in parse_csv_arg(args.seeds)]

    for dataset in datasets:
        for seed in seeds:
            dm = PolarisDatamodule(
                dataset=dataset,
                seed=seed,
                batch_size=32,
                featurizer=None,
                use_benchmark=True,
                save_benchmark_dataset=True,
            )
            dm.prepare_benchmark_split_cache()

    stamp_path = Path(args.stamp_path)
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    stamp_path.write_text(
        f"prepared datasets={','.join(datasets)} seeds={','.join(map(str, seeds))}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
