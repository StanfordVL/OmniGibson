Hi folks, the data & tools are now ready for the fillable annotation pass! Please read the below information in detail prior to starting.

**Reminder, our goal: finish this week!**

## Setting Up

1. Download a copy of the latest, textureless, dataset [HERE](https://storage.googleapis.com/gibson_scenes/fillable-10-21.zip) and extract this at some path on your drive.
2. Download the script [HERE](https://github.com/StanfordVL/ig_pipeline/blob/main/b1k_pipeline/usd_conversion/view_fillable_volumes.py).
3. Open the script and edit the gm.DATASET_PATH line to point at the extraction location from the previous step.
4. Make sure you are on `og-develop` branch of OmniGibson
5. Make sure you are on the `develop` branch of BDDL, and make sure that your environment does not use a pip-installed BDDL. The script should warn you if this is the case.
6. Install pyglet: `pip install "pyglet<2"`
6. As with before, go to the spreadsheet [HERE](https://docs.google.com/spreadsheets/d/10L8wjNDvr1XYMMHas4IYYP9ZK7TfQHu--Kzoi0qhAe4/edit?gid=1388270730) and pick a batch and put your name on it. Then run `python view_fillable_volumes.py BATCH_ID 20 potato` (BATCH_ID is the batch number you picked, and yes you need to type potato).

## IMPORTANT: HOW TO PICK / Requirements

* You can skip a mesh if you want - you SHOULD do this to skip items that cannot get good fillable meshes with the current tooling, so that we can revisit them.

* These are the requirements of a fillable mesh:
    * It should cover as much of the object's cavity as possible, and should be in near contact with the bottom. If it is too far from covering the whole cavity, it will cause issues with filledness detection.
    * **It should **NOT** intersect/go through the object in any way**
    * It should end near the top of the cavity, but it can be slightly above or slightly below.

* One key part of the fillable annotation is **where** the annotation is, that is, whether the annotation will move with the base link (e.g. in a washing machine) or with one of the links (e.g. in a drawer unit). All of the pre-generated meshes move with the base link, as such, they should NOT be selected for objects where the fillable volume should move with a link, such objects should use a generated mesh.

## The Process 

We will be doing QA/selection of pregenerated options and annotation of alternatives in a single pass. With each object, decide if you want to pick one of the three pre-generated meshes, or generate a new one manually.

* Generating fillable meshes is simple:
  1. You start by picking the link you will add the annotation to using the link selection keys. The display will update to show just that link and its descendants.
  2. Then you can use the 1D arrows or the 2D squares to move the SEED prim using the Isaac Sim interface to be roughly at the center of the volume you want to annotate. **Do NOT move directly in all three axes since it will just go somewhere you don't want it to.**
  3. When ready, you have two options: you can hit K or L to do the ray cast (PLEASE continue reading the next item to learn when to use which). If it works, you will get a green fillable mesh. If not, a dialog may pop up showing you why the ray cast failed. A message will also be printed.
  4. If you don't like the result, you can remove it by pressing O.
  5. You can repeat this to add as many fillable volumes as you want to each link, and you can add volumes to as many links as you want, with the exception that a link may only have a fillable volume if none of its ancestors or descendants have one.
  6. When done, you can hit Z to save all the generated meshes and record the generated meshes as the selected option.

## Enclosed vs Open Meshes
There are two options for generating fillable meshes: the "enclosed" option (by pressing K) and the "open" option (by pressing L). The difference is that, in the enclosed option, the rays must actually hit a part of the object, requiring and guaranteeing that the annotated location is somewhat enclosed. In the open option, rays are allowed to hit the object's convex hull too, guaranteeing a mesh, but these may be overapproximations, and since this method will also work with areas that are not properly enclosed, volumes generated this way will be flagged so that we don't use them for sampling fluids etc. - as a result, you should refrain from using the OPEN mode **unless** absolutely necessary e.g. in the case of shelf units.

## Key bindings

* Press J to skip
* Press U to indicate we should remove the fillable annotation from the object.
* Press X to choose the dip (red) option.
* Press S to choose the ray (blue) option.
* Press W to choose the combined (purple) option.
* Press V to toggle visibility of the visual meshes (off by default)
* Press C to toggle visibility of the collision meshes (on by default)
* Press A to select the next link.
* Press Q to select the previous link.
* Press K to generate an ENCLOSED fillable mesh from the current seed point.
* Press L to generate an OPEN fillable mesh from the current seed point, allowing convex hull hits.
* Press O to remove the last generated mesh.
* Press P to clear all generated meshes.
* Press Z to pick the generated option.

## Finishing up
At the end of the run, I will send you a script that will package the generated meshes in a zip file. For now, make sure you do NOT delete or overwrite the dataset directory.