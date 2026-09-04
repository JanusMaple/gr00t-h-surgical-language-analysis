"""Reconstruct the GR00T-H N1.7 Surgical training-language mixture.

The 57-entry training recipe defines the extraction scope. Public Open-H rows
provide within-group language proportions, while the configured group ratios
provide training-mixture weights. Missing historical datasets are reported and
never imputed. Existing N1.7 count output is reused as a cache unless ``--rescan``
is supplied.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import csv
import json
import re
import sys

import pandas as pd
import pyarrow.parquet as pq


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

DATA_ROOT = Path("open_h_subset/Surgical")
GR00T_ROOT = Path("gr00t_h").resolve()
TRAINING_YAML = GR00T_ROOT / "open_h" / "gr00t_h_config.yaml"
OUTPUT_DIR = Path("extracted_language")
COUNTS_CSV = OUTPUT_DIR / "gr00t_h_n17_public_language_counts.csv"

sys.path.insert(0, str(GR00T_ROOT))

# Prompt constructors used by the GR00T-H preprocessing pipeline.
from open_h.embodiments.cmr_versius.utils.cmr_add_state_prompts import (  # noqa: E402
    PROCEDURE_MAP,
    build_prompt as build_cmr_prompt,
)
from open_h.embodiments.rob_surgical_bitrack.utils.rob_surgical_add_tool_prompts import (  # noqa: E402
    build_augmented_instruction as build_rob_prompt,
)


# -----------------------------------------------------------------------------
# Authoritative N1.7 training groups + current-public mappings
# -----------------------------------------------------------------------------

# Each local_path is relative to open_h_subset/Surgical.
# local_path=None means exact language cannot currently be reconstructed.
TRAINING_GROUPS = [
    {
        "group": "cmr_versius",
        "mix_ratio": 1.6000,
        "mix_percent": 19.20,
        "expected_frames": 105_945_933,
        "exclude_splits": [],
        "entries": [
            ("cmr-surgical-60hz/cholecystectomy_phase_gesture_prompts_v1", "cmr_surgical/cholecystectomy", "CMR_STATE_PROMPT"),
            ("cmr-surgical-60hz/hysterectomy_phase_prompts_v1", "cmr_surgical/hysterectomy", "CMR_STATE_PROMPT"),
            ("cmr-surgical-60hz/inguinal_hernia", "cmr_surgical/inguinal_hernia", "CMR_STATE_PROMPT"),
            ("cmr-surgical-60hz/prostatectomy", "cmr_surgical/prostatectomy", "CMR_STATE_PROMPT"),
        ],
    },
    {
        "group": "jhu_imerse_dvrk",
        "mix_ratio": 2.9144,
        "mix_percent": 34.98,
        "expected_frames": 5_269_530,
        "exclude_splits": ["missing_videos"],
        "entries": [
            ("Surgical/JHU/LSCR/ARCADE/Cholecystectomy", "jhu/lcsr/arcade/cholecystectomy", "TASK_INDEX"),
            ("Surgical/JHU/LSCR/ARCADE/cautery", "jhu/lcsr/arcade/cautery", "TASK_INDEX"),
            ("Surgical/JHU/srth_porcine_chole_fix", "jhu/imerse/srth_porcine_chole", "TASK_INDEX"),
            ("Surgical/JHU/cao_cautery_combined", "jhu/imerse/cao_cautery_combined", "TASK_INDEX"),
            ("Surgical/JHU/suturebot_tissue_2", None, "ASSUMED_SUTUREBOT_FAMILY"),
            ("Surgical/JHU/srt_needle_pickup+handover", "jhu/imerse/srt_needle_pickup_handover", "TASK_INDEX"),
            ("Surgical/JHU/suturing_Jesse_processed_fix", None, "ASSUMED_SUTUREBOT_FAMILY"),
            ("Surgical/JHU/srt_tissue_lift", "jhu/imerse/srt_tissue_lift", "TASK_INDEX"),
            ("Surgical/JHU/jesse_pickup_only", None, "ASSUMED_SUTUREBOT_FAMILY"),
            (
                "Surgical/JHU/Imerse/Wound_Closure/point_labeled",
                "jhu/imerse/wound_closure/point_labeled/fausto_0_1_jesse_0_1_2_labeled",
                "TASK_INDEX",
            ),
            ("Surgical/JHU/suturebot", "jhu/imerse/suturebot", "TASK_INDEX"),
            ("Surgical/JHU/Imerse/NephFat_extracted/nephfat", "jhu/imerse/nephfat/nephfat", "TASK_INDEX"),
        ],
    },
    {
        "group": "jhu_imerse_dvrk_mono",
        "mix_ratio": 0.3102,
        "mix_percent": 3.72,
        "expected_frames": 516_334,
        "exclude_splits": [],
        "entries": [
            ("Surgical/JHU/suturebot", "jhu/imerse/suturebot", "TASK_INDEX"),
        ],
    },
    {
        "group": "jhu_lscr_dvrk_smarts",
        "mix_ratio": 0.0619,
        "mix_percent": 0.74,
        "expected_frames": 103_025,
        "exclude_splits": [],
        "entries": [
            (
                "Surgical/JHU/LSCR/SMARTS/offline_recorder_extracted/offline_data_part1",
                "jhu/lcsr/smarts/SurgSync-stitch-coldcut/P1",
                "TASK_INDEX",
            ),
            (
                "Surgical/JHU/LSCR/SMARTS/offline_recorder_extracted/offline_data_part2",
                "jhu/lcsr/smarts/SurgSync-stitch-coldcut/P2",
                "TASK_INDEX",
            ),
            (
                "Surgical/JHU/LSCR/SMARTS/offline_recorder_extracted/offline_data_part3",
                "jhu/lcsr/smarts/SurgSync-stitch-coldcut/P3",
                "TASK_INDEX",
            ),
        ],
    },
    {
        "group": "stanford_dvrk_real",
        "mix_ratio": 0.5253,
        "mix_percent": 6.30,
        "expected_frames": 874_437,
        "exclude_splits": ["fail", "bad_frames"],
        "entries": [
            (
                "Surgical/Stanford/Collaborative Haptics and Robotics in Medicine Lab/Real Robot (dVRK)/Needle Transfer",
                "stanford/collaborative_haptics_and_robotics_in_medicine_lab/real_robot_dvrk/needle_transfer",
                "TASK_INDEX",
            ),
            (
                "Surgical/Stanford/Collaborative Haptics and Robotics in Medicine Lab/Real Robot (dVRK)/Tissue Retraction",
                "stanford/collaborative_haptics_and_robotics_in_medicine_lab/real_robot_dvrk/tissue_retraction",
                "TASK_INDEX",
            ),
            (
                "Surgical/Stanford/Collaborative Haptics and Robotics in Medicine Lab/Real Robot (dVRK)/Peg Transfer",
                "stanford/collaborative_haptics_and_robotics_in_medicine_lab/real_robot_dvrk/peg_transfer",
                "TASK_INDEX",
            ),
        ],
    },
    {
        "group": "obuda_dvrk",
        "mix_ratio": 0.6949,
        "mix_percent": 8.34,
        "expected_frames": 1_156_946,
        "exclude_splits": [],
        "entries": [
            ("Surgical/Obuda/FRS_Dome_1", "obuda/frs_dome_1", "TASK_INDEX"),
            ("Surgical/Obuda/NeedleThreading_1", "obuda/needlethreading_1", "TASK_INDEX"),
            ("Surgical/Obuda/PegTransfer_1", "obuda/pegtransfer_1", "TASK_INDEX"),
            ("Surgical/Obuda/Rollercoaster_1", "obuda/rollercoaster_1", "TASK_INDEX"),
            ("Surgical/Obuda/Seaspike_1", "obuda/seaspike_1", "TASK_INDEX"),
            ("Surgical/Obuda/NeedleThreading_2", "obuda/needlethreading_2", "TASK_INDEX"),
            ("Surgical/Obuda/PegTransfer_2", "obuda/pegtransfer_2", "TASK_INDEX"),
            ("Surgical/Obuda/Pork_1", "obuda/pork_1", "TASK_INDEX"),
            ("Surgical/Obuda/Seaspike_2", "obuda/seaspike_2", "TASK_INDEX"),
            ("Surgical/Obuda/Seaspike_3", "obuda/seaspike_3", "TASK_INDEX"),
            ("Surgical/Obuda/Skinphantom_1", "obuda/skinphantom_1", "TASK_INDEX"),
        ],
    },
    {
        "group": "rob_surgical_bitrack",
        "mix_ratio": 0.6031,
        "mix_percent": 7.24,
        "expected_frames": 1_003_887,
        "exclude_splits": [],
        "entries": [
            ("Surgical/Rob Surgical/all_merged_data", "rob_surgical/all_merged_data", "ROB_TOOL_PROMPT"),
        ],
    },
    {
        "group": "jhu_imerse_star_il",
        "mix_ratio": 0.0704,
        "mix_percent": 0.85,
        "expected_frames": 117_247,
        "exclude_splits": ["MISSING_VIDEOS"],
        "entries": [
            ("Surgical/JHU/Imerse/star_IL_extracted/star_IL", "jhu/imerse/star_il/star_il", "TASK_INDEX"),
        ],
    },
    {
        "group": "ustc_torin_tuodao",
        "mix_ratio": 0.3075,
        "mix_percent": 3.69,
        "expected_frames": 512_030,
        "exclude_splits": [],
        "entries": [
            ("Surgical/USTC/exvivo_liver_sep", None, "MISSING_DIRECT_TEXT"),
            ("Surgical/USTC/grasp_on_liver", None, "MISSING_DIRECT_TEXT"),
            ("Surgical/USTC/invivo_liver_sep", None, "MISSING_DIRECT_TEXT"),
            ("Surgical/USTC/ustc_knot_tying", None, "MISSING_DIRECT_TEXT"),
            ("Surgical/USTC/Needle_handover", None, "MISSING_DIRECT_TEXT"),
            ("Surgical/USTC/needle_pickup", None, "MISSING_DIRECT_TEXT"),
            ("Surgical/USTC/tissue_lifting", None, "MISSING_DIRECT_TEXT"),
        ],
    },
    {
        "group": "hamlyn_dvrk_30hz",
        "mix_ratio": 0.3272,
        "mix_percent": 3.93,
        "expected_frames": 544_573,
        "exclude_splits": ["failure"],
        "entries": [
            ("Surgical/Hamlyn/Suturing-2", "hamlyn/suturing_2", "TASK_INDEX"),
            ("Surgical/Hamlyn/peg_transfer", "hamlyn/peg_transfer", "TASK_INDEX"),
            ("Surgical/Hamlyn/Suturing-1", "hamlyn/suturing_1", "TASK_INDEX"),
            ("Surgical/Hamlyn/needle_grasp_and_handover", "hamlyn/needle_grasp_and_handover", "TASK_INDEX"),
            ("Surgical/Hamlyn/knot_tying", "hamlyn/knot_tying", "TASK_INDEX"),
            ("Surgical/Hamlyn/Tissue_Retraction", "hamlyn/tissue_retraction", "TASK_INDEX"),
        ],
    },
    {
        "group": "ucsd_dvrk",
        "mix_ratio": 0.1892,
        "mix_percent": 2.27,
        "expected_frames": 314_917,
        "exclude_splits": [],
        "entries": [
            ("Surgical/UCSD/surgical_learning_dataset", "ucsd/surgical_learning_dataset", "TASK_INDEX"),
            ("Surgical/UCSD/surgical_learning_dataset2", "ucsd/surgical_learning_dataset2", "TASK_INDEX"),
        ],
    },
    {
        "group": "ucb_dvrk",
        "mix_ratio": 0.1333,
        "mix_percent": 1.60,
        "expected_frames": 221_950,
        "exclude_splits": [],
        "entries": [
            ("Surgical/UCBerkeley/debridement_lerobot", "ucberkeley/debridement_lerobot", "TASK_INDEX"),
        ],
    },
    {
        "group": "turin_mitic_ex_vivo",
        "mix_ratio": 0.5661,
        "mix_percent": 6.80,
        "expected_frames": 997_835,
        "exclude_splits": ["failure", "MISSING_VIDEOS"],
        "entries": [
            ("Surgical/Turin/mitic_lerobot_ex_vivo", "turin/mitic_lerobot_ex_vivo", "DIRECT_TEXT:instruction.text"),
            ("Surgical/Turin/mitic_lerobot_plastic_pad", "turin/mitic_lerobot_plastic_pad", "DIRECT_TEXT:instruction.text"),
            ("Surgical/Turin/mitic_lerobot_plastic_pad_3DMED", "turin/mitic_lerobot_plastic_pad_3dmed", "DIRECT_TEXT:instruction.text"),
            ("Surgical/Turin/mitic_lerobot_plastic_tube", "turin/mitic_lerobot_plastic_tube", "DIRECT_TEXT:instruction.text"),
        ],
    },
    {
        "group": "tud_tundra_ur5e",
        "mix_ratio": 0.0287,
        "mix_percent": 0.34,
        "expected_frames": 47_753,
        "exclude_splits": [],
        "entries": [
            ("Surgical/TUD/260131_TUNDRA_dataset/grasping_retraction", "tud/260131_tundra_dataset/grasping_retraction", "TASK_INDEX"),
        ],
    },
]

ASSUMED_SUTUREBOT_VOCABULARY = ["knot tying", "needle pickup", "needle throw"]


# -----------------------------------------------------------------------------
# Training-recipe validation
# -----------------------------------------------------------------------------

def validate_training_yaml() -> None:
    if not TRAINING_YAML.exists():
        print(f"WARNING: training YAML not found: {TRAINING_YAML}")
        return

    text = TRAINING_YAML.read_text(encoding="utf-8")
    paths = re.findall(r"^\s*-\s+\$\{OPEN_H_DATA_PATH\}/([^\n#]+)", text, flags=re.MULTILINE)
    paths = [p.strip() for p in paths]

    print(f"Training YAML entries:       {len(paths)}")
    print(f"Unique historical paths:     {len(set(paths))}")

    duplicates = sorted(p for p in set(paths) if paths.count(p) > 1)
    print(f"Duplicated historical paths: {duplicates}")

    if len(paths) != 57 or len(set(paths)) != 56:
        print("WARNING: current YAML no longer matches the investigated 57-entry / 56-unique recipe.")


# -----------------------------------------------------------------------------
# Extraction helpers
# -----------------------------------------------------------------------------

def get_parquets(dataset_root: Path) -> list[Path]:
    return sorted(dataset_root.glob("data/**/*.parquet"))


def load_task_map(dataset_root: Path) -> dict[int, str]:
    path = dataset_root / "meta" / "tasks.jsonl"
    mapping: dict[int, str] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            idx = int(obj["task_index"])
            task = str(obj.get("task", ""))
            if idx in mapping and mapping[idx] != task:
                raise RuntimeError(f"Conflicting task_index {idx} in {path}")
            mapping[idx] = task

    return mapping


def normalize_task_index(value, parquet_path: Path) -> int:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Expected one task_index in {parquet_path}, got {value!r}")
        value = value[0]
    return int(value)


def read_column(parquet_path: Path, column: str):
    pf = pq.ParquetFile(parquet_path)
    if column not in pf.schema_arrow.names:
        raise KeyError(f"{column!r} missing from {parquet_path}")
    return pf.read(columns=[column])[column].to_pylist()


def extract_task_index_language(dataset_root: Path) -> tuple[Counter, Counter]:
    task_map = load_task_map(dataset_root)
    counts = Counter()
    unresolved = Counter()

    parquets = get_parquets(dataset_root)
    for i, parquet in enumerate(parquets, 1):
        for value in read_column(parquet, "task_index"):
            if value is None:
                continue
            idx = normalize_task_index(value, parquet)
            if idx in task_map:
                counts[task_map[idx]] += 1
            else:
                unresolved[idx] += 1
        if i % 1000 == 0:
            print(f"    {i:,}/{len(parquets):,} files")

    return counts, unresolved


def extract_direct_text(dataset_root: Path, column: str) -> Counter:
    counts = Counter()
    parquets = get_parquets(dataset_root)

    for i, parquet in enumerate(parquets, 1):
        for value in read_column(parquet, column):
            text = "" if value is None else str(value).strip()
            counts[text] += 1
        if i % 1000 == 0:
            print(f"    {i:,}/{len(parquets):,} files")

    return counts


def extract_cmr(dataset_root: Path) -> Counter:
    procedure_name = dataset_root.name
    procedure = PROCEDURE_MAP[procedure_name]
    counts = Counter()
    parquets = get_parquets(dataset_root)

    for i, parquet in enumerate(parquets, 1):
        for state in read_column(parquet, "observation.state"):
            counts[build_cmr_prompt(state, procedure)] += 1
        if i % 1000 == 0:
            print(f"    {i:,}/{len(parquets):,} files")

    return counts


def extract_rob(dataset_root: Path) -> Counter:
    counts = Counter()
    required = [
        "instruction.text",
        "observation.meta.left_tool",
        "observation.meta.right_tool",
        "observation.meta.aux_tool",
    ]

    for parquet in get_parquets(dataset_root):
        df = pd.read_parquet(parquet, columns=required)
        for _, row in df.iterrows():
            counts[build_rob_prompt(row)] += 1

    return counts


def extract_mechanism(dataset_root: Path, mechanism: str) -> tuple[Counter, Counter]:
    unresolved = Counter()

    if mechanism == "CMR_STATE_PROMPT":
        return extract_cmr(dataset_root), unresolved
    if mechanism == "ROB_TOOL_PROMPT":
        return extract_rob(dataset_root), unresolved
    if mechanism == "TASK_INDEX":
        return extract_task_index_language(dataset_root)
    if mechanism.startswith("DIRECT_TEXT:"):
        column = mechanism.split(":", 1)[1]
        return extract_direct_text(dataset_root, column), unresolved

    raise ValueError(f"Cannot extract mechanism: {mechanism}")


# -----------------------------------------------------------------------------
# Cached N1.7 counts
# -----------------------------------------------------------------------------

def load_cached_counts() -> dict[tuple[str, str], Counter]:
    """Load unique dataset/mechanism counts from a previous N1.7 run."""
    if not COUNTS_CSV.exists():
        return {}

    required = {
        "current_public_path",
        "mechanism",
        "language",
        "raw_public_count",
    }
    out: dict[tuple[str, str], Counter] = {}
    with COUNTS_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            print(f"WARNING: {COUNTS_CSV} is missing {sorted(missing)}; ignoring it.")
            return {}

        for line_no, row in enumerate(reader, 2):
            key = (row["current_public_path"], row["mechanism"])
            language = row["language"]
            try:
                count = int(row["raw_public_count"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid cached count at {COUNTS_CSV}:{line_no}") from exc

            counter = out.setdefault(key, Counter())
            existing = counter.get(language)
            if existing is not None and existing != count:
                raise ValueError(
                    f"Conflicting cached counts for {key!r}, {language!r}: "
                    f"{existing} and {count}"
                )
            counter[language] = count

    print(f"Reusing N1.7 language counts for {len(out)} dataset/mechanism pairs.")
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rescan",
        action="store_true",
        help=(
            "Ignore existing N1.7 public-language counts and re-read mapped Parquets."
        ),
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    validate_training_yaml()

    total_manifest_entries = sum(len(g["entries"]) for g in TRAINING_GROUPS)
    print(f"Manifest training entries:   {total_manifest_entries}")
    assert total_manifest_entries == 57

    prior_counts = {} if args.rescan else load_cached_counts()
    cache: dict[tuple[str, str], tuple[Counter, Counter]] = {}

    count_rows = []
    summary_rows = []
    missing_rows = []
    group_rows = []
    inventory_sources: dict[str, set[str]] = {}

    for group in TRAINING_GROUPS:
        group_name = group["group"]
        group_public_frames = 0
        mapped_entries = 0
        missing_entries = 0

        print("=" * 110)
        print(
            f"GROUP {group_name}: expected={group['expected_frames']:,} frames, "
            f"mix={group['mix_percent']:.2f}%"
        )

        for entry_i, (historical_path, local_path, mechanism) in enumerate(group["entries"]):
            entry_id = f"{group_name}:{entry_i:02d}"

            if local_path is None:
                missing_entries += 1
                if mechanism == "ASSUMED_SUTUREBOT_FAMILY":
                    note = (
                        "Historical JHU/SutureBot-family training subset absent under this name in current public Open-H. "
                        "Reasonable lexical assumption: task vocabulary is a subset of {knot tying, needle pickup, needle throw}; "
                        "exact per-task frame counts are unknown and are not fabricated."
                    )
                    assumed_vocab = " | ".join(ASSUMED_SUTUREBOT_VOCABULARY)
                else:
                    note = (
                        "Confirmed USTC GR00T-H training entry absent from current public Open-H. "
                        "Language was per-frame instruction.text; exact strings unavailable."
                    )
                    assumed_vocab = ""

                missing_rows.append(
                    {
                        "entry_id": entry_id,
                        "group": group_name,
                        "historical_path": historical_path,
                        "mechanism": mechanism,
                        "mix_percent_group": group["mix_percent"],
                        "note": note,
                        "assumed_vocabulary_for_lexical_context_only": assumed_vocab,
                    }
                )
                print(f"  MISSING  {historical_path}")
                continue

            mapped_entries += 1
            root = DATA_ROOT / local_path
            if not root.exists():
                raise FileNotFoundError(f"Mapped local dataset does not exist: {root}")

            key = (local_path, mechanism)
            unresolved = Counter()

            if key in prior_counts:
                counts = prior_counts[key].copy()
                source = "n17_counts_cache"
            elif key in cache:
                counts, unresolved = cache[key]
                counts = counts.copy()
                unresolved = unresolved.copy()
                source = "cached_current_run"
            else:
                print(f"  SCAN     {historical_path} -> {local_path} [{mechanism}]")
                counts, unresolved = extract_mechanism(root, mechanism)
                cache[key] = (counts.copy(), unresolved.copy())
                source = "parquet_scan"

            raw_frames = sum(counts.values())
            group_public_frames += raw_frames

            for text, count in counts.items():
                count_rows.append(
                    {
                        "entry_id": entry_id,
                        "group": group_name,
                        "historical_training_path": historical_path,
                        "current_public_path": local_path,
                        "mechanism": mechanism,
                        "mix_ratio_group": group["mix_ratio"],
                        "mix_percent_group": group["mix_percent"],
                        "language": text,
                        "raw_public_count": count,
                        "count_semantics": "current_public_raw_rows_not_exact_training_probability",
                    }
                )
                inventory_sources.setdefault(text, set()).add(entry_id)

            summary_rows.append(
                {
                    "entry_id": entry_id,
                    "group": group_name,
                    "historical_training_path": historical_path,
                    "current_public_path": local_path,
                    "mechanism": mechanism,
                    "source": source,
                    "raw_public_rows": raw_frames,
                    "unique_prompts": len(counts),
                    "unresolved_task_index_rows": sum(unresolved.values()),
                    "group_exclude_splits_in_training_yaml": " | ".join(group["exclude_splits"]),
                }
            )

        expected = group["expected_frames"]
        delta = group_public_frames - expected
        group_rows.append(
            {
                "group": group_name,
                "mix_ratio": group["mix_ratio"],
                "mix_percent": group["mix_percent"],
                "expected_training_frames_yaml": expected,
                "current_public_raw_rows_mapped_entries": group_public_frames,
                "raw_minus_expected": delta,
                "mapped_entries": mapped_entries,
                "missing_entries": missing_entries,
                "exclude_splits_in_training_yaml": " | ".join(group["exclude_splits"]),
                "warning": (
                    "Raw current-public rows are not exact training rows when historical snapshots differ, "
                    "entries are missing, or exclude_splits were used."
                ),
            }
        )

    # -------------------------------------------------------------------------
    # Convert raw public counts into training-mixture weights
    # -------------------------------------------------------------------------

    total_mix_ratio = sum(float(group["mix_ratio"]) for group in TRAINING_GROUPS)
    group_public_totals = {
        row["group"]: int(row["current_public_raw_rows_mapped_entries"])
        for row in group_rows
    }
    entry_public_totals = {
        row["entry_id"]: int(row["raw_public_rows"])
        for row in summary_rows
    }

    represented_groups = {
        group["group"]
        for group in TRAINING_GROUPS
        if group_public_totals[group["group"]] > 0
    }
    represented_mixture_weight = sum(
        float(group["mix_ratio"]) / total_mix_ratio
        for group in TRAINING_GROUPS
        if group["group"] in represented_groups
    )

    weighted_language: dict[str, dict] = {}

    for row in count_rows:
        group_total = group_public_totals[row["group"]]
        entry_total = entry_public_totals[row["entry_id"]]
        group_weight = float(row["mix_ratio_group"]) / total_mix_ratio
        entry_weight = entry_total / group_total
        language_within_entry = int(row["raw_public_count"]) / entry_total
        mixture_weight = group_weight * entry_weight * language_within_entry

        row["group_mixture_weight"] = group_weight
        row["entry_within_group_weight"] = entry_weight
        row["language_within_entry_weight"] = language_within_entry
        row["training_mixture_weight"] = mixture_weight
        row["normalized_reconstructed_weight"] = (
            mixture_weight / represented_mixture_weight
        )

        aggregate = weighted_language.setdefault(
            row["language"],
            {
                "raw_public_count": 0,
                "training_mixture_weight": 0.0,
                "groups": set(),
                "entry_ids": set(),
            },
        )
        aggregate["raw_public_count"] += int(row["raw_public_count"])
        aggregate["training_mixture_weight"] += mixture_weight
        aggregate["groups"].add(row["group"])
        aggregate["entry_ids"].add(row["entry_id"])

    weighted_rows = []
    for language, aggregate in weighted_language.items():
        mixture_weight = aggregate["training_mixture_weight"]
        weighted_rows.append(
            {
                "language": language,
                "normalized_training_mixture_weight": (
                    mixture_weight / represented_mixture_weight
                ),
                "training_mixture_weight_before_coverage_normalization": mixture_weight,
                "training_mixture_percent_before_coverage_normalization": 100 * mixture_weight,
                "raw_public_count": aggregate["raw_public_count"],
                "num_training_entries": len(aggregate["entry_ids"]),
                "groups": " | ".join(sorted(aggregate["groups"])),
                "entry_ids": " | ".join(sorted(aggregate["entry_ids"])),
                "represented_training_mixture_weight": represented_mixture_weight,
                "weight_semantics": (
                    "YAML group mix ratios normalized globally; language proportions "
                    "estimated from mapped current-public rows within each represented group"
                ),
            }
        )

    weighted_rows.sort(
        key=lambda row: row["normalized_training_mixture_weight"],
        reverse=True,
    )

    normalized_weight_sum = sum(
        row["normalized_training_mixture_weight"] for row in weighted_rows
    )
    if abs(normalized_weight_sum - 1.0) > 1e-9:
        raise RuntimeError(
            f"Normalized language weights sum to {normalized_weight_sum}, not 1"
        )

    for row in group_rows:
        row["normalized_training_mixture_weight"] = (
            float(row["mix_ratio"]) / total_mix_ratio
        )
        row["language_distribution_reconstructed"] = (
            row["group"] in represented_groups
        )

    # -------------------------------------------------------------------------
    # Write outputs
    # -------------------------------------------------------------------------

    def write_dict_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    counts_path = COUNTS_CSV
    summary_path = OUTPUT_DIR / "gr00t_h_n17_public_language_summary.csv"
    missing_path = OUTPUT_DIR / "gr00t_h_n17_missing_training_entries.csv"
    coverage_path = OUTPUT_DIR / "gr00t_h_n17_group_coverage.csv"
    inventory_path = OUTPUT_DIR / "gr00t_h_n17_unique_language_inventory.csv"
    weighted_path = OUTPUT_DIR / "gr00t_h_n17_weighted_language.csv"
    notes_path = OUTPUT_DIR / "gr00t_h_n17_reconstruction_notes.txt"

    write_dict_csv(counts_path, count_rows)
    write_dict_csv(summary_path, summary_rows)
    write_dict_csv(missing_path, missing_rows)
    write_dict_csv(coverage_path, group_rows)
    write_dict_csv(weighted_path, weighted_rows)

    inventory_rows = [
        {
            "language": text,
            "num_training_entries_containing_string": len(entries),
            "entry_ids": " | ".join(sorted(entries)),
        }
        for text, entries in sorted(inventory_sources.items())
    ]
    write_dict_csv(inventory_path, inventory_rows)

    with notes_path.open("w", encoding="utf-8") as f:
        f.write(
            "GR00T-H N1.7 language reconstruction summary\n"
            "==========================================\n\n"
            "Training recipe:\n"
            "- 57 training entries in gr00t_h_config.yaml\n"
            "- 56 unique historical paths\n"
            "- Surgical/JHU/suturebot appears twice under different embodiment configs\n"
            "- YAML summary comment says 58 datasets, but actual path list has 57\n\n"
            "Resolved mechanisms:\n"
            "- CMR: NVIDIA instruction.text_with_state reconstruction; N1.6->N1.7 semantics unchanged\n"
            "- Rob Surgical: NVIDIA instruction.text_with_tool reconstruction\n"
            "- Turin: raw instruction.text\n"
            "- Most other mapped entries: task_index -> tasks.jsonl\n\n"
            "Known gaps:\n"
            "- 7 USTC entries (3.69% group mix): current public data absent; exact instruction.text unavailable\n"
            "- 3 JHU historical SutureBot-family entries: exact public mapping unavailable\n"
            "  Assumption for lexical interpretation only: vocabulary likely subset of {knot tying, needle pickup, needle throw}\n"
            "  Exact frequency is not fabricated.\n\n"
            "Important frequency caveat:\n"
            "- normalized_training_mixture_weight applies the exact normalized YAML group mix ratios.\n"
            "- Within each represented group, language proportions are estimated from mapped current-public rows.\n"
            "- episode_sampling_rate=0.1 is common to all datasets and therefore cancels from relative weights.\n"
            "- Some historical snapshots and exclude_splits cannot be reconstructed exactly from current public data.\n"
            "- USTC has no reconstructed language, so normalized_reconstructed_weight conditions on represented groups.\n"
            f"- Represented mixture mass before that final normalization: {represented_mixture_weight:.12f}.\n"
        )

    print("\n" + "=" * 110)
    print("DONE")
    print(f"Mapped training entries:  {len(summary_rows)}")
    print(f"Missing training entries: {len(missing_rows)}")
    print(f"Unique reconstructed strings: {len(inventory_rows):,}")
    print(f"Represented training-mixture mass: {represented_mixture_weight:.6%}")
    print(f"Normalized language-weight sum: {normalized_weight_sum:.12f}")
    print("\nSaved:")
    print(f"  {counts_path}")
    print(f"  {summary_path}")
    print(f"  {missing_path}")
    print(f"  {coverage_path}")
    print(f"  {inventory_path}")
    print(f"  {weighted_path}")
    print(f"  {notes_path}")


if __name__ == "__main__":
    main()
