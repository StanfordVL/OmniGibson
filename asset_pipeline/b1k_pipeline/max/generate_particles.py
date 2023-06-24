import pymxs
rt = pymxs.runtime

import csv
import string
import random

def generate_particle(name, shape, colors, size):
    for color in colors.split(";"):
        if shape == "cube":
            obj = rt.Box()
            obj.width = float(size)
            obj.height = float(size)
            obj.length = float(size)
        elif shape == "sphere":
            obj = rt.Sphere()
            obj.radius = float(size) / 2
        else:
            raise ValueError("Unknown shape " + shape) 
            
        # Assign a name
        model_id = "".join(random.choices(string.ascii_lowercase, k=6))
        obj.name = f"{name}-{model_id}-0"
        
        # Create the material
        r, g, b = [int(float(x) * 255) for x in color.split(",")]
        mat = rt.VRayMtl()
        mat.diffuseColor = rt.Color(r, g, b)
        print("Created", obj.name)
        return obj
        
def main():
    with open(r"D:\ig_pipeline\metadata\substance_generation.csv") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if row["Request"] != "GENERATE" or row["Shape"] not in ("cube", "sphere"):
                continue
            particle = generate_particle(row["Name"], row["Shape"], row["Color"], row["Diameter (mm)"])
            particle.position = rt.Point3(i * 100, 0, 0)
            
if __name__ == "__main__":
    main()