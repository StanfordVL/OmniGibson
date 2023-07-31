import os
import pathlib
import gspread
import csv

OUTPUT_ROOT = pathlib.Path(__file__).parents[1] / "generated_data"

ASSETS_SHEET_KEY = "10L8wjNDvr1XYMMHas4IYYP9ZK7TfQHu--Kzoi0qhAe4"
SYNSETS_SHEET_KEY = "1eIQn1HzUJV15nCP4MqsHvrdWAV9VrKoxOqSnQxF0_1A"
SYMSET_PARAMS_SHEET_KEY = "1jQomcQS3DSMLctEOElafCPj_v498gwtb57NC9eNwu-4"

ALL_SHEETS = [
  (ASSETS_SHEET_KEY, "Object Category Mapping", "category_mapping.csv"),
  (ASSETS_SHEET_KEY, "Allowed Room Types", "allowed_room_types.csv"),
  (ASSETS_SHEET_KEY, "Substance Hyperparams", "substance_hyperparams.csv"),
  (SYNSETS_SHEET_KEY, "Synsets", "synsets.csv"),
  (SYMSET_PARAMS_SHEET_KEY, "heatsource", "prop_param_annots/heatSource.csv"),
  (SYMSET_PARAMS_SHEET_KEY, "coldsource", "prop_param_annots/coldSource.csv"),
  (SYMSET_PARAMS_SHEET_KEY, "cookable", "prop_param_annots/cookable.csv"),
  (SYMSET_PARAMS_SHEET_KEY, "flammable", "prop_param_annots/flammable.csv"),
  (SYMSET_PARAMS_SHEET_KEY, "heatable", "prop_param_annots/heatable.csv"),
  (SYMSET_PARAMS_SHEET_KEY, "particleApplier", "prop_param_annots/particleApplier.csv"),
  (SYMSET_PARAMS_SHEET_KEY, "particleSource", "prop_param_annots/particleSource.csv"),
  (SYMSET_PARAMS_SHEET_KEY, "particleRemover", "prop_param_annots/particleRemover.csv"),
  (SYMSET_PARAMS_SHEET_KEY, "particleSink", "prop_param_annots/particleSink.csv")
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