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

### View Complaints
The View Complaints tool shows you unresolved complaints that are attached to the currently selected object, or if no object is selected, it shows you the complaints on all the objects in the current file.

### Resolve Complaints
The Resolve Complaints tool marks as resolved all of the complaints shown in View Complaints for the selected object. It should be used after all work on the object is completed and confirmed.

### Merge Collision
The Merge Collision tool allows you to quickly attach, triangulate, and rename a set of meshes to be assigned as the collision mesh of an object.

% TODO: More needed here!

### Convex Decomposition
The Convex Decomposition tool runs some off-the-shelf convex decomposition algorithms (mainly CoACD and V-HACD) to generate collision mesh candidates for a selected object.

% TODO: More needed here!

### Convex Hull
The Convex Hull tool computes the convex hull of the selected mesh, or if an element of the mesh is selected, of the element.

% TODO: More needed here!

### Validate Collision
The validate collision tool checks that the selected collision mesh satisfies all of the requirements: naming, element count, and vertex count. It's a simplified, fast version of the collision mesh checks inside the file-level sanitycheck.

## Steps to Follow

Quick note: if any of the steps below fail due to missing dependencies, you can install them by opening the `Scripting > Maxscript Listener` dialog, clicking `Python`, and running e.g. `import pip; pip.main(["install", "coacd"])` where `coacd` is the package you want to install. To run the line you need to press Shift + Enter.

After reading all of the above, follow the below setup steps:

1. `git pull` the ig_pipeline repository's `main` branch
2. Create your own branch for what you're working on, e.g. `git checkout -b collision`
2. `dvc pull` in the ig_pipeline repository to make sure you have the latest files.
3. **Make sure you did BOTH of the above! Otherwise you will be editing an outdated copy of the file and you'll have to redo the work because these files cannot be diffed and merged.**
4. "Unprotect" all of the files. This allows you to edit the files, replacing the symlinks with editable copies of the file. You can run this Powershell command to do that: % TODO

After doing the above, follow the steps described below for each file matching cad/objects/*/processed.max (e.g. in batch-00)

1. Press `View Complaints` to show all complained objects in the file.
2. Search for, and select, the first object. Note that the complaint will be at object level, howwever, if the object complaints multiple links, the problem might be on any one of the links, or all of them.
    * To search for an object, type/paste its 6 character model ID into the search box in the Scene Explorer, and put an asterisk (`*`) before the ID.
    * If you type the asterisk first, the application will likely freeze as you type/paste the rest - not recommended!
3. Carefully evaluate, and understand, what the problem might be that a complaint was added. Review the `Understanding the Requirements` section if needed.
4. Correct the issue. You can do one of the below:
    1. Create a collision mesh from scratch. To do this, remove the existing collision mesh by deleting the Mcollision object altogether. Then use the Convex Decomposition etc. tools to carefully generate a new collision mesh. % TODO
    2. Make edits to the existing collision mesh
5. Make sure you have merged all parts of the collision mesh of each link into a single object, suffixed Mcollision and placed underneath the visual object in the Scene Explorer tree. Use the `Merge Collision` tool to do this. How to use it is documented above.
6. Select the collision mesh and use the `Validate Collision` tool to validate the collision mesh. Confirm in the MaxScript Listener dialog that the output is VALID.
7. If **any link of** the object has a fillable volume also annotated (e.g. a Mfillable object underneath the link) - **you MUST remove it and recreate it!** Delete the Mfillable mesh and follow the steps in the `fillable.md` file in this directory for a description of how to create one.
8. Mark the complaint as resolved by selecting the object and running the `Resolve Complaint` tool.
9. Save the file to make sure you don't lose work.
10. Repeat from Step 1 until there are no complaints left in that file.
11. Save the file once more.
12. Run `dvc add [filename]` where filename is the name of the file you just edited, e.g. `dvc add cad/objects/batch-00/processed.max`. This will make DVC cache the file and update the pointer file to point at your edited version.
13. `git commit` the edited file and also the complaints.json file in the same directory.

To upload your work (e.g. when done, or if you want us to check progress):

1. Run `dvc push` and make sure it finishes
2. Run `git push` to your branch on the ig_pipeline repository. **Make sure you did step 1 before this, otherwise you'll be pushing just pointers and not actual files!**
3. Open a PR on the ig_pipeline repository and tag @cgokmen for review.