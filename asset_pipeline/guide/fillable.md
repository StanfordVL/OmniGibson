# Fillable Volumes

## Understanding the Requirements

### Open or Closed?
There are two options for generating fillable meshes: the default, "enclosed" option and the "open" option. The difference is that, in the enclosed option, the rays must actually hit a part of the object, requiring and guaranteeing that the annotated location is somewhat enclosed. In the open option, rays are allowed to hit the object's convex hull too, guaranteeing a mesh, but these may be overapproximations, and since this method will also work with areas that are not properly enclosed, volumes generated this way will be flagged so that we don't use them for sampling fluids etc. - as a result, you should refrain from using the OPEN mode unless absolutely necessary e.g. in the case of shelf units.

## Understanding the Tooling

## Steps to Follow
Pre-steps: see if the object actually has an accessible cavity (e.g. the collision mesh doesnt block it). If it doesn't, we might want to fix the object if it's the only one of its kind (see knowledgebase website), or if not, we can delete the object.

## Option 1: the fillable volume generator
1. Pick the link you are doing this for. You need to pick the link that the fillable volume will move with. For example, a cabinet with doors will have its thing on the base link. A drawer unit will have a separate one on each of its drawers. Your object might be a combination of these (e.g. both the base link AND the drawers will have volumes)
2. Use the add fillable seed button to add the seed
3. Move the seed to roughly the middle of the cavity (use the 1d axes or 2d planes on the move tool, dont move in 3d directly. makes it hard)
4. Click the Generate Fillable Volume or Generate Open Fillable Volume button (more on this below)
Keep doing this until the entire cavity is covered.

## Option 2: doing it manually
1. If the generator won't generate a mesh, if the object has a shape that's suitable for it, you can manually create the mesh using convex primitives like box, cone etc. Make sure you don't over or underapproximate, and most importantly, that you don't include any external volume as part of the fillable volume. 
2. After creating the collision mesh out of primitive shapes, use the `Merge Collision` tool to merge the volumes, and make sure you rename the final object to Mfillable