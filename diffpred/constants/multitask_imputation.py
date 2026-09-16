MANIFEST_SCHEMA_VERSION = "1.0"

MASK_MAP_SPLIT_COL = "split"
MASK_MAP_SAMPLE_ID_COL = "sample_id"
MASK_MAP_ROW_ID_COL = "row_id"
MASK_MAP_TARGET_COL = "target_col"
MASK_MAP_IS_MASKED_COL = "is_masked_for_input"
MASK_MAP_TRUE_VALUE_COL = "y_true_original"
MASK_MAP_RATIO_COL = "mask_ratio"
MASK_MAP_SEED_COL = "seed"

MASK_MAP_REQUIRED_COLUMNS = [
    MASK_MAP_SPLIT_COL,
    MASK_MAP_SAMPLE_ID_COL,
    MASK_MAP_ROW_ID_COL,
    MASK_MAP_TARGET_COL,
    MASK_MAP_IS_MASKED_COL,
    MASK_MAP_TRUE_VALUE_COL,
    MASK_MAP_RATIO_COL,
    MASK_MAP_SEED_COL,
]

MANIFEST_REQUIRED_TOP_LEVEL_KEYS = [
    "schema_version",
    "seed",
    "split_policy",
    "target_columns",
    "splits",
]

MANIFEST_REQUIRED_SPLITS = ["train", "val", "test"]
