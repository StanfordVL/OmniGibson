import pymxs
rt = pymxs.runtime

import sys
sys.path.append(r"D:\ig_pipeline")

import b1k_pipeline.utils

import csv
import string
import random
from PIL import Image, ImageDraw, ImageFont
import textwrap

def generate_texture(label, path):
    # create an image
    out = Image.new("RGB", (1000, 500), (255, 255, 255))
    HEIGHT = out.height
    WIDTH = out.width

    lines = textwrap.wrap(label, 12)
    assert len(lines) <= 3
    text = "\n".join(lines)
    FONT_SIZE = 172
    fnt = ImageFont.truetype("impact.ttf", FONT_SIZE, encoding="unic")
    d = ImageDraw.Draw(out)
    d.multiline_text((WIDTH // 2, HEIGHT // 2), text, font=fnt, anchor="mm", align="center", fill=(30, 30, 30))

    out.save(path)

def generate_container(cat, label, base_model):
    # Find the stuff belonging to the base
    base_objs = [x for x in rt.objects if x.name.contains(base_model)]
    base_tree = {x for x in rt.objects if x in base_objs or x.parent in base_objs}
    assert base_tree, f"Could not find any objects with ID {base_model}"

    # Compile the basename for replacement
    base_cat = b1k_pipeline.utils.parse_name(base_objs[0]).group("category")
    base_name = f"{base_cat}-{base_model}"

    # What will this object be called?


    objs = []
    # Generate the object
    obj = get_shape(shape, size)
        
    # Assign a name
    model_id = "".join(random.choices(string.ascii_lowercase, k=6))
    obj.name = f"{name}-{model_id}-0"

    # Create the collision version
    col_obj = get_shape(shape, size)
    col_obj.name = obj.name + "-Mcollision"
    col_obj.parent = obj
    col_obj.isHidden = True
    
    # Create the material
    r, g, b = [int(x) for x in color.split(",")]
    mat = rt.VRayMtl()
    mat.diffuse = rt.Color(r, g, b)
    obj.material = mat

    objs.append(obj)
    print("Created", obj.name)
    return objs
        
def main():
    with open(r"D:\ig_pipeline\metadata\container_generaetion.csv") as f:
        containers = []
        for row in csv.DictReader(f):
            if row["Request"] != "GENERATE":
                continue
            containers.extend(generate_container(row["Category"], row["Label"], row["Actual Object"]))

        for i, container in enumerate(containers):
            container.position = rt.Point3(i * 100, 0, 0)
        
if __name__ == "__main__":
    main()