import pymxs
rt = pymxs.runtime

import csv
import string
import random

def get_shape(shape, size):
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
    
    return obj

def generate_particle(name, shape, colors, size):
    objs = []
    for color in colors.split(";"):
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
    with open(r"D:\ig_pipeline\metadata\substance_generation.csv") as f:
        particles = []
        for row in csv.DictReader(f):
            if row["Request"] != "GENERATE" or row["Shape"] not in ("cube", "sphere"):
                continue
            particles.extend(generate_particle(row["Name"], row["Shape"], row["Color"], row["Diameter (mm)"]))

        for i, particle in enumerate(particles):
            particle.position = rt.Point3(i * 100, 0, 0)
            
if __name__ == "__main__":
    main()