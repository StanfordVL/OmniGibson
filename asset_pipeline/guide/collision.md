# Collision Issues
This guide describes how to address collision/handle complaints.

## Understanding the Requirements
Every object in the BEHAVIOR-1K dataset must be assigned a collision mesh. 

* The collision mesh is the union of a set of convex volumes
* A collision mesh can consist of at most 40 convex volumes
* Each convex volume must have at most 60 vertices
* The different convex volumes comprising a collision mesh are allowed to intersect
one another. This does not cause any issues and can be used to simplify meshes
for gridlike objects.
* The collision mesh must completely represent the object's affordances with the smallest
possible number of convex volumes. For example:
    * Cavities of fillable objects (e.g. bottles, etc.) must NOT be closed off
    * Supporting surfaces e.g. of shelves, cabinets, etc. must be correctly represented: they
    should be flat, correctly sloped
    * Handles should be clearly represented, and if they have a gap underneath e.g. where a finger would normally fit, such gaps should NOT be closed off

**If an object is assigned a collision complaint, its current collision mesh was deemed by a reviewer to have issues with one or more of the above requirements.** It is very important to correctly identify the issue prior to starting work on the object.

## Understanding the Tooling
The ig_pipeline repository contains some tools to allow quick processing of collision meshes. These can be installed by running the `b1k_pipeline/add_buttons.py` script from the Scripting > Run Script menu dialog. After running this script a SVL menu will be placed in the main menu of 3ds Max with buttons for each of the below described scripts and more.

### Merge Collision
The Merge Collision tool allows you to quickly attach, triangulate, and rename a set of meshes to be assigned as the collision mesh of an object.

% TODO: More needed here!

### Convex Decomposition
The Convex Decomposition tool runs some off-the-shelf convex decomposition algorithms (mainly CoACD and V-HACD) to generate collision mesh candidates for a selected object.

% TODO: More needed here!

### Convex Hull
The Convex Hull tool computes the convex hull of the selected mesh 