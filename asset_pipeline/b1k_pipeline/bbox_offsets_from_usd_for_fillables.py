from concurrent.futures import ProcessPoolExecutor
import json
import os
import pathlib
import tempfile
from cryptography.fernet import Fernet
import pxr.Usd
import numpy as np
from tqdm import tqdm

FILLABLE_DIR = pathlib.Path(r"D:\fillable-10-21")
KEY_PATH = pathlib.Path(r"C:\Users\cgokmen\research\OmniGibson\omnigibson\data\omnigibson.key")


def decrypt_file(encrypted_filename, decrypted_filename):
    with open(KEY_PATH, "rb") as filekey:
        key = filekey.read()
    fernet = Fernet(key)

    with open(encrypted_filename, "rb") as enc_f:
        encrypted = enc_f.read()

    decrypted = fernet.decrypt(encrypted)

    with open(decrypted_filename, "wb") as decrypted_file:
        decrypted_file.write(decrypted)


def get_bounding_box_from_usd(input_usd, tempdir):
    encrypted_filename = input_usd
    fd, decrypted_filename = tempfile.mkstemp(suffix=".usd", dir=tempdir)
    os.close(fd)
    decrypt_file(encrypted_filename, decrypted_filename)
    stage = pxr.Usd.Stage.Open(str(decrypted_filename))
    prim = stage.GetDefaultPrim()

    base_link_size = np.array(prim.GetAttribute("ig:nativeBB").Get()) * 1000
    base_link_offset = np.array(prim.GetAttribute("ig:offsetBaseLink").Get()) * 1000

    return base_link_offset, base_link_size

def main():
    input_usds = list(FILLABLE_DIR.glob("objects/*/*/usd/*.usd"))
    print(len(input_usds))

    # Scale up
    with tempfile.TemporaryDirectory() as tempdir:
      with ProcessPoolExecutor() as executor:
          results = list(tqdm(executor.map(get_bounding_box_from_usd, input_usds, [pathlib.Path(tempdir)] * len(input_usds)), total=len(input_usds)))

    json_friendly_names = [usd.parts[-3] for usd in input_usds]
    json_friendly_results = [(offset.tolist(), size.tolist()) for offset, size in results]
    with open(FILLABLE_DIR / "fillable_bboxes.json", "w") as f:
        json.dump(dict(zip(json_friendly_names, json_friendly_results)), f)

if __name__ == "__main__":
    main()