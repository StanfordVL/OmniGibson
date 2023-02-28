import sys
sys.path.append(r"D:\ig_pipeline")

import csv
import json
import traceback

import b1k_pipeline.utils

PIPELINE_OUTPUT_DIR = b1k_pipeline.utils.PIPELINE_ROOT / "artifacts" / "pipeline"
OBJECT_INVENTORY_PATH = PIPELINE_OUTPUT_DIR / "object_inventory.json"
OUTPUT_FILENAME = PIPELINE_OUTPUT_DIR / "aggregate_metadata.json"
SUCCESS_FILENAME = PIPELINE_OUTPUT_DIR / "aggregate_metadata.success"

CATEGORY_MAPPING_FILENAME = b1k_pipeline.utils.PIPELINE_ROOT / "metadata" / "category_mapping.csv"

METADATA_ROOT_DIR = b1k_pipeline.utils.PIPELINE_ROOT / "artifacts" / "aggregate" / "metadata"
CATEGORIES_FILENAME = METADATA_ROOT_DIR / "categories.txt"

def main():
    METADATA_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    success = True
    error_msg = ""
    try:
        with open(OBJECT_INVENTORY_PATH, "r") as f:
            object_inventory = json.load(f)
    
        assert object_inventory["success"], "Object inventory was unsuccessful."

        # categories = [obj.split("-")[0] for obj in object_inventory["providers"].keys()]
        
        # For now, get categories from CSV file
        categories_by_id = {}
        with open(CATEGORY_MAPPING_FILENAME, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                cat_id = int(row["id"].strip())
                category = row["category"].strip()
                categories_by_id[cat_id] = category
        
        # Fill missing IDs with spaces
        categories = [
            categories_by_id[i] if i in categories_by_id else ""
            for i in range(max(categories_by_id.keys()) + 1)
        ]

        categories_w_newline = [x + "\n" for x in categories]
        with open(CATEGORIES_FILENAME, "w") as f:
            f.writelines(categories_w_newline)
    except Exception as e:
        success = False
        error_msg = traceback.format_exc()

    with open(OUTPUT_FILENAME, "w") as f:
        json.dump({"success": success, "error_msg": error_msg}, f, indent=4)

if __name__ == "__main__":
    main()