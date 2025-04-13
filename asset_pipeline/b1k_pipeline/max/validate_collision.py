from pymxs import runtime as rt
import trimesh
import numpy as np

def validate_collision_mesh(obj, max_elements=40, max_vertices_per_element=60):
    # For this case, unwrap the object
    obj = obj._obj

    # Expect that collision meshes do not share instances in the scene
    self.expect(
        not [
            x for x in rt.objects if x.baseObject == obj.baseObject and x != obj
        ],
        f"{obj.name} should not have instances.",
    )

    # Check that there are no dead elements
    self.expect(
        rt.polyop.GetHasDeadStructs(obj) == 0,
        f"{obj.name} has dead structs. Apply the Triangulate script.",
    )

    # Get vertices and faces into numpy arrays for conversion
    verts = self.get_verts_for_obj(obj)
    faces = self.get_faces_for_obj(obj)
    self.expect(len(faces) > 0, f"{obj.name} has no faces.")
    self.expect(
        all(len(f) == 3 for f in faces),
        f"{obj.name} has non-triangular faces. Apply the Triangulate script.",
    )

    all_cmeshes = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # Split the faces into elements
    elems = all_cmeshes.split(only_watertight=False, repair=False)
    if max_elements is not None:
        self.expect(
            len(elems) <= max_elements,
            f"{obj.name} should not have more than {max_elements} elements. Has {len(elems)} elements.",
        )

    # Iterate through the elements
    for i, m in enumerate(elems):
        if max_vertices_per_element is not None:
            self.expect(
                len(m.vertices) <= max_vertices_per_element,
                f"{obj.name} element {i} has too many vertices ({len(m.vertices)} > {max_vertices_per_element})",
            )
        self.expect(m.is_volume, f"{obj.name} element {i} is not a volume")
        self.expect(m.is_convex, f"{obj.name} element {i} is not convex")

if __name__ == "__main__":
    validate_collision_mesh(rt.selection[0])