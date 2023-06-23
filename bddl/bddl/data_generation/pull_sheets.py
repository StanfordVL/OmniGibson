import os
import pathlib
import gspread
import csv

OUTPUT_ROOT = pathlib.Path(__file__).parents[1] / "generated_data"

ASSETS_SHEET_KEY = "10L8wjNDvr1XYMMHas4IYYP9ZK7TfQHu--Kzoi0qhAe4"
SYNSETS_SHEET_KEY = "1eIQn1HzUJV15nCP4MqsHvrdWAV9VrKoxOqSnQxF0_1A"
SYMSET_PARAMS_SHEET_KEY = "1GXXa0uAsXiULVMELxrhFNNgjC_cBHXvj0uxT571Jnzs"
TM_SHEET_KEY = "11PsOlRYjsJ_WZZzMc6-iOi0oT_pIgKgvzXLg3kR00KM"

ALL_SHEETS = [
  (ASSETS_SHEET_KEY, "Object Category Mapping", "category_mapping.csv"),
  (ASSETS_SHEET_KEY, "Allowed Room Types", "allowed_room_types.csv"),
  (SYNSETS_SHEET_KEY, "Synsets", "synsets.csv"),
  (SYMSET_PARAMS_SHEET_KEY, "heatsource", "prop_param_annots/heatSource.csv"),
  (SYMSET_PARAMS_SHEET_KEY, "coldsource", "prop_param_annots/coldSource.csv"),
  (SYMSET_PARAMS_SHEET_KEY, "cooking", "prop_param_annots/cooking.csv"),
  (TM_SHEET_KEY, "cleaning substance-based dirtiness", "prop_param_annots/tm_cleaning.csv"),
]

def main():
  OUTPUT_ROOT.mkdir(exist_ok=True)
  client = gspread.service_account(filename=os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
  for key, sheet_name, filename in ALL_SHEETS:
    worksheet = client.open_by_key(key).worksheet(sheet_name)
    with open(OUTPUT_ROOT / filename, "w") as f:
      writer = csv.writer(f)
      writer.writerows(worksheet.get_all_values())

if __name__ == "__main__":
  main()