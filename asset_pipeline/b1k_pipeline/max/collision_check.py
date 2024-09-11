import pymxs
rt = pymxs.runtime

obj = rt.selection[0]

# Expect that collision meshes do not share instances in the scene
assert not [x for x in rt.objects if x.baseObject == obj.baseObject and x != obj], f"{obj.name} should not have instances."

# Check that there are no dead elements
assert rt.polyop.GetHasDeadStructs(obj) == 0, f"{obj.name} has dead structs. Apply the Triangulate script."

# Get vertices and faces into numpy arrays for conversion
verts = np.array([rt.polyop.getVert(obj, i + 1) for i in range(rt.polyop.GetNumVerts(obj))])
faces_maxscript = [rt.polyop.getFaceVerts(obj, i + 1) for i in range(rt.polyop.GetNumFaces(obj))]
faces = np.array([[int(v) - 1 for v in f] for f in faces_maxscript if f is not None])
assert all(len(f) == 3 for f in faces), f"{obj.name} has non-triangular faces. Apply the Triangulate script."

import collections
edges_seen = collections.Counter()
for face in faces:
    a, b, c = face
    edges_seen[frozenset([a, b])] += 1
    edges_seen[frozenset([b, c])] += 1
    edges_seen[frozenset([c, a])] += 1
    
print(edges_seen)
    
for edge, cnt in edges_seen.items():
    if cnt != 2:
        print("Edge", edge, "seen", cnt, "times")

# Split the faces into elements
raw_elems = {tuple(rt.polyop.GetElementsUsingFace(obj, i + 1)) for i in range(rt.polyop.GetNumFaces(obj))}
assert len(raw_elems) <= 32, f"{obj.name} should not have more than 32 elements. Has {len(elems)} elements."
elems = np.array(list(raw_elems))
assert not np.any(np.sum(elems, axis=0) > 1), f"{obj.name} has same face appear in multiple elements"

print(elems)

print(len(elems), "elems")

# Check that no face goes through the same vertex twice
for i, face in enumerate(faces):
    assert len(set(face)) == len(face), face

# Iterate through the elements
for i, elem in enumerate(elems):
    # Load the mesh into trimesh and expect convexity
    relevant_faces = faces[elem]
    m = trimesh.Trimesh(vertices=verts, faces=relevant_faces, process=False)
    m.remove_unreferenced_vertices()
    
    import networkx as nx
    G = nx.Graph()
    for face in m.faces:
        a, b, c = face
        G.add_edge(a, b)
        G.add_edge(b, c)
        G.add_edge(c, a)
        
    print(G)
        
    for edge, cnt in edges_seen2.items():
        if cnt != 2:
            print("Edge", edge, "seen", cnt, "times")
    
    if not m.is_volume:
        rt.polyop.setFaceSelection(obj, (np.argwhere(elem).flatten() + 1).tolist())
    assert m.is_volume, f"{obj.name} element {i} is not a volume"
    # assert (m.is_convex, f"{obj.name} element {i} is not convex")
    assert len(m.split()) == 1, f"{obj.name} element {i} has elements trimesh still finds splittable"