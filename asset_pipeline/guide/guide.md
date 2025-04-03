# **B1K Scene & Object Annotation Guide**

# A. Things To Learn How To Do

This section consists of a list of things you should know how to do. Watch the videos first and refer back as needed.

If you think you prefer a general tutorial covering some of the below stuff and general familiarity with 3ds Max, take a look at this [50-min video](https://www.youtube.com/watch?v=YM9spHSNPpM). We have also curated a set of shorter videos for things that we think are important:

1. Intro to the 3ds Max interface: [7min video](https://www.youtube.com/watch?v=D7LaYg5-pB0)  
2. Moving / rotating camera: [4min video](https://www.youtube.com/watch?v=yZcJGej-pf0)  
   1. We recommend that you set up your viewports such that you have 4 viewports: perspective, top, one of (left, right), one of (front, back). You can switch the “one of” options as needed. Use the viewport cube to easily rotate the viewports.  
   2. Our scenes are large and complicated, and thus it’s hard to work with the left/right/front/back views since many objects will be stacked. To help with this, we recommend the use of Viewport Clipping, which lets you define two planes to act as bounds of the rendering area (e.g. anything that’s not between the two parallel planes will not be rendered). This way you can see slices of the scene cleanly. [See here](https://knowledge.autodesk.com/support/3ds-max/learn-explore/caas/sfdcarticles/sfdcarticles/Viewport-Clipping.html) for more info.  
3. Moving / rotating objects: [6min video](https://www.youtube.com/watch?v=0Mz56Br2tIw)  
4. Different selection modes (edge / border / element): You will need to use these especially when splitting / joining / etc objects. [Brief read](https://knowledge.autodesk.com/support/3ds-max/learn-explore/caas/CloudHelp/cloudhelp/2021/ENU/3DSMax-Modeling/files/GUID-409D36AC-1D17-4B24-851D-1C49C7E0B79D-htm.html).  
5. Merging objects: [5min video](https://www.youtube.com/watch?v=0iSsUfzxy6o)  
6. Splitting objects (when the object already consists of separate elements): [1min video](https://www.youtube.com/watch?v=eTbIs1vcaWY)   
7. Cutting objects (with a slice plane): [3min video](https://www.youtube.com/watch?v=Z8Z_BM5Z6Uw)  
8. Closing gaps: select the border of the gap, either using the Border or the Edge selection mode. Then click on the Cap button if in the Border mode or the Bridge button if in the Edge mode.  
9. Setting pivots: [2min video](https://www.youtube.com/watch?v=Gd2B29CeZEg)  
10. TODO: Using the snap-to-grid settings

# B. Scene / Object Annotation Process

## 0\. Starting

First, move into a branch: run \`git checkout \-b scene-name\` where scene-name is the name of the scene you will be working on. Mark the scene name in the [spreadsheet](https://docs.google.com/spreadsheets/d/10L8wjNDvr1XYMMHas4IYYP9ZK7TfQHu--Kzoi0qhAe4/edit?usp=sharing) and track your progress.

Since we are sharing these computers with other students, make sure you use the correct branch (your branch\!) before you start working, and also save progress (save, git commit, etc) at all time.

We don’t want to overwrite the raw scene. As soon as you open it, click Save As in the File menu and save it as processed.max in the same directory. **You should dvc add and then commit your file after \*every\* numbered step below.**

## 1\. Apply required scene modifications

Any required modifications to the scene need to be applied first. This involves ensuring all of the following are true:

* Any scene combination/edits have been completed in advance. For the B-1K dataset scenes, please chat with area leads for any particular requirements for your particular scene.  
* The scene overall contains a natural boundary (e.g. walls) that is properly closed.  
* The scene floor is flat.  
* We should only include objects that are useful for our environment. E.g. if an object seems like it should be usable (e.g. a stack of pans) but cannot be used (because you cannot unstack them), we should remove this object.

## 2\. Check if all objects are properly segmented

Check that each “object” is correctly identified as an object in the object list. This means you need to be able to select in the object list all the parts of that object. Some rules:

* Every object needs to be matched to an object category in the iGibson object [category list](https://docs.google.com/spreadsheets/d/1JJob97Ovsv9HP1Xrs_LYPlTnJaumR2eMELImGykD22A/edit#gid=2141444742) (“Object Category B1K” tab, the “object” column) and be interchangeable with all other objects in the same category. If the right category exists for all of the objects in your scene, great, use that. Otherwise, talk to Cem and Eric about adding the categories. You can look inside the [iGibson dataset](https://github.com/StanfordVL/ig_dataset/tree/master/objects) online to see the other objects that fall into that category, to check if your object is roughly interchangeable.   
  * If you end up creating a new category, please find an appropriate [WordNet synset](http://wordnetweb.princeton.edu/perl/webwn) for it and add it to the spreadsheet.   
* Consider object randomization: one of the goals of iGibson is to be able to replace any object of a given type (e.g. “sink”) with any other object of that same type.  
* Consider movement: any moving parts (e.g. a door frame vs the door leaf) should be separated  
* Consider repetition: if there are 5 stalls in a restroom, they should be annotated as 5 objects rather than 1 object covering all 5\.

To achieve this segmentation state, some objects might need to be split into parts (e.g. object already has the parts as separate “elements” and you just need to move the elements into their own objects) or cut (e.g. the parts are by default a single mesh that needs to be cut), and others might need to be combined. Use your knowledge from Part B to do this. You might also need to fill gaps.

### **Special Case: Walls, Floors and Ceilings**

There are some specific requirements about the annotations of walls, floors and ceilings that need to be addressed.

Each wall object needs to consist of a flat wall segment, e.g. there should not be any corners within the object. This is important because the tool we use to convert visual meshes to collision, VHACD, has a tendency to cut corners within each object, causing in unreachable areas and collision with scene objects. **Wall objects currently do not need to be assigned into a room.**

Floor and ceiling objects are expected to be watertight meshes, with a thickness of at least 30cm, and floors need to have a flat surface at a height of Z=0. Ideal floor meshes should consist of very few vertices (\<20, exactly 8 for a cuboid shape, etc). This will mean some floors may need to be remeshed from scratch. **Floors and ceilings need to be assigned to rooms, and each room should have its own floors. The floor objects will be used to compute the room map as well as checking what room the robot is in.**

## 3\. Perform vertex reduction

One of the goals of iGibson is to be able to provide fast simulation. To be able to do this, we need our objects to not be too complex. The mechanism to check for this is built into sanitycheck: no objects will be allowed to have more than 20,000 vertices.

If an object has more than 20,000 vertices, you need to apply vertex reduction. To do this, select the object, go to the modifiers tab, and from the modifier dropdown add the ProOptimizer modifier.

You can try values in different ranges (try 500, 1000, 5000, 10000 and 20000). Ideally we want to pick the lowest of the above options were the object still looks reasonably good. If in doubt, ask for help from area leads.

Don’t forget to merge in this modifier as part of the below step.

## 4\. Flatten modifiers, convert to Editable Poly

3ds Max supports multiple different types of objects (‘editable mesh’, ‘editable poly’, etc.) as well as “modifiers” on objects that dynamically change their shape and can be reverted. For our pipeline to work, all objects need to be of type editable poly, and all modifiers need to be merged into the mesh (e.g. “flattened”). For each object whose modifier stack does not consist of exactly a single “Editable Poly” entry, follow the below steps.

**Important: Doing this using the method explained below will allow instances of your object to stay instances rather than being disconnected, reducing the total amount of work you will need to do later.**

* Go into the modifiers tab on the right-hand panel (2nd tab).  
* Using the dropdown, add an Edit Poly modifier.  
* Make sure that the Edit Poly modifier is added to the top of the modifier stack.  
* Right click on the Edit Poly modifier and click Collapse To.  
* If a warning dialog pops up, click Yes.  
* Confirm on the Scene Explorer dialog (left-hand side menu) that instances of your object are still instances.

![][image1.png]        ![][image2.png]

## 5\. Replace object copies with instances

When there are multiple copies of the same object models, we naturally want to export this model once, and use the same model for each of its instances, rather than making multiple copies. To do this, 3ds Max needs to recognize that the repeated copies are instances of the main object and not new objects themselves. Such objects are known as “instances”.

Start by enabling the Display \> Display Dependents option in the object list:  
![][image3.png]

Now, individually click on every object that is used multiple times in the scene. If the copies are correctly marked as “instances”, they will also be highlighted in the list once you select any instance of the same object. If that does not happen, the multiple objects are copies and not instances, and they need to be replaced.

To do this, use the Clone-and-align tool (Tools \> Align \> Clone and Align). Pivots need to be in the same position and in the same orientation across the different objects (relative to their own object model) in order for the clone and align tool to work properly. Select the object you want to use as the main copy, then use the Pick List option to pick all of the clones to replace. Use the “Instance” mode and hit clone when ready. Close the tool, verify that in the list, the instances are correctly highlighted now, and one by one, remove the original clones (paying attention to make sure that the new instance matches the removed clone’s location exactly).

You can also use the provided Clone-and-Align script to complete this process.

## 6\. Annotate moving parts (“links”)

Many objects are “articulated”, e.g. they are not a single rigid body but consist of multiple rigid parts that are allowed to move relative to each other. For example, a door is an articulated object with a moving door leaf, a drawer unit is an articulated object with moving drawers, etc. \- support for such articulation is what puts the i in iGibson (“interactive”). To be able to simulate these articulations, iGibson needs some information about the joint connecting the different parts: the origin, the axis, the limits, etc. \- normally quite hard to annotate. Fortunately, our pipeline allows us to compute all of this information automatically \- all it needs is each articulated joint’s lower and upper positions.

By this point, for each articulated object, you should have its fixed and moving parts as separate objects, as requested for part 2\. Then, for each such object, do the following:

1. Move the moving part so that it’s exactly at the “closed” position of the joint. Make a mental note that this is the “lower” position of the joint, for naming later.  
2. Make a copy of the object using the Clone tool in the Edit menu (choose Instance as the cloning type).  
3. Decide if the joint needs to be a prismatic (e.g. linear / translational) or revolute (e.g. rotational) joint. Note that it should be one of the two: in the below steps, you should *either* translate or rotate, *but not both*.  
   1. For a prismatic joint, simply move the copy to its final (“upper”) position using your knowledge from Part B. Take care to make sure that the two objects are moved only along the desired joint axis, and not along any of the other two lateral axes.  
   2. For a revolute joint, you will need to rotate the copy to the final (“upper”) position. To rotate an object:  
      1. You first need to set a pivot around which you will perform the rotation. Use your knowledge from Part B to set this (temporary) pivot. Pick the pivot in such a way that the moving part will not collide with the fixed part, not only in its starting/final positions, but also along the way. For example, see in the below figure the right position and a few wrong positions for the pivot.  
         ![][image4.png]  
      2. Then, using the rotate tool, rotate the moving part around the pivot. You can manually enter the amount (degrees) to rotate for many revolute joints, e.g. 90 degrees, so that the rotation is exact. See the “exact transforms” FAQ tip. **You should not apply a rotation of more than 180 degrees: in these cases, the URDF joint will follow the minor arc, producing undesirable results.**

## 7\. Annotate local coordinate frames

In iGibson, objects by default are expected to have their local coordinate frames set up such that they have the X axis (red arrow) facing forward, and the Z axis (blue arrow) facing upward. The third (Y) axis also needs to be in the correct position according to the right-hand rule (Z x X \= Y). Our next step involves making sure this is the case for all of our objects. In 3ds Max, we use the “Pivot” feature of each object to mark its local coordinate frame.

For each object in the scene:

* Select the object. If you are doing this for an object where multiple instances exist, you can select all of the instances. For articulated object parts, select both the lower and upper copies, and annotate with the closed position in mind.  
* Go into the Pivot options \- these are in the right-hand toolbar, in the Hierarchy menu.  
* Enable the Affect Pivot Only option.  
* Click the Center To Object and Align To Object options to move the coordinate frame origin to a reasonable starting point.  
* Then use the rotating tool so that the axes satisfy the above requirements.

## 8\. Rename objects

Below is our naming scheme. Split each token with a \-:

* If the object has a bad model (e.g. a cabinet that has some fake drawers) \- start with the prefix B-. Note that this means you also need to provide a replacement object model ID from the dataset \- see the model ID part below.  
* If the object does not lend itself well to randomization (e.g. hangers with different spaces between their hooks, console tables with different height shelves where things need to go) \- add prefix F- so that they are fixed and not randomized.  
* If the object is loose (e.g. not a fixed object in this scene but a movable one) \- add prefix L-  
  * This only applies to base links \- e.g. the fact that a door leaf is moving doesn’t require this since the door frame (the base link) does not move.  
  * Typically in iGibson, furniture objects that you’d expect to be fixed IRL are marked as fixed, and smaller objects (e.g. soap dispenser, garbage can, etc.) are marked as “loose”. If you’re not sure if an object should be fixed or loose, talk to us.   
* The object category, where separate words are separated using underscore (\_) characters. For example: bottom\_cabinet.  
* The object model ID, a six-character lowercase ASCII string that should be unique to this model in the dataset. If there are multiple different models of objects of the same category (e.g. not multiple instances of the same model but different models altogether), they should get different model IDs. If the object is a bad model, the suggested replacement object’s ID (from any of the CAD files from the scene) should be inserted here.  
* The object instance number, 0 by default. If there are multiple instances of the same model (e.g. multiple copies of the same toilet), they should have different numbers here, preferably in some reasonable order.  
* The link name, if this link is not the base link of this object.  
* The parent link name, if this link is a moving link. If the parent is the base link of the object, use base\_link.  
* Joint type: P for prismatic (e.g. the object moves linearly), R for revolute (e.g. the object rotates around an axis), **New:** F for fixed (e.g. the object does not move relative to its parent)  
* Joint side: “upper” if this instance marks the upper end of the joint, “lower” if it marks the lower end. By convention, we choose the lower end to be the end that keeps the object “closed”. **New:** For fixed joints, only a lower side should be specified.  
* **New:** Meta link type (e.g. “Mwatersource”), should be alphanumeric with NO UNDERSCORES. (this is relevant only for meta links, see part 10\)  
* **New:** Optional meta link id (e.g. “left”), should be alphanumeric with NO UNDERSCORES. (this is relevant only for meta links, see part 10\)  
* **New:** Optional meta link sub-id (e.g. “0”), should be alphanumeric with NO UNDERSCORES. (this is relevant only for meta links, see part 10\)  
* **New:** Optional tags (multiple allowed), \-Ttagname1-Ttagname2 etc. (see Tags section below)

Some examples:

* B-L-bottom\_cabinet-0-0  
* L-soap\_dispenser-0-0  
* Sink-0-1  
* door-0-0-base\_link  
* door-0-0-leaf-base\_link-R-lower  
* door-0-0-handle-leaf-R-lower  
* cabinet-0-0-drawer-base\_link-P-lower-Topenable  
* F-sink-0-0-Mfluidsource\_left

### **NEW: Tags**

Tags are a new mechanism that allow us to add metadata to individual objects. Multiple tags can be added onto each object by appending the letter T followed by the tag name. For example, to apply the glass tag, \-Tglass can be added to an object.

Below are the currently supported tags:

* **Tsoft:** the object needs to be simulated as a soft body rather than a rigid body (e.g. cloth, towels, etc.). **Note that any object that has this tag should come in a default-unfolded shape.**  
* **Tglass:** the object needs to be simulated as glass rather than using its current material. Note that this will apply to the entire object, so any object that is only partially glass needs to be segmented into separate links (with fixed joints) so that the entire thing doesn’t turn into glass.  
* **Tlocked:** when placed on the base link of a scene object, the object’s articulations will be locked in their lower position in that scene. Prevents robots from falling out of unlocked doors.  
* **Topenable:** on an object with multiple links, this link should be taken into account when computing the “openable” state (e.g. it’s not an irrelevant link like buttons etc.)  
* **Topenable\_both\_sides:** on an openable object, the links are set up such that if they are both fully open, the object is semantically “closed”. This is mostly useful for things like sliding doors and windows, and is only helpful for two-pane ones and not more.  
* **Tsubpart:** indicates a part/whole relationship where the part replaces the whole, e.g. when an object is sliced. Is only valid on an object that has another object as its parent. See part 10 for more details.  
* **Textrapart:** indicates a part/whole relationship where the part should be added to the scene in addition to the whole, e.g. the pillow on a bed, **but is not connected to the whole with a joint (e.g. they will move separately)**. Is only valid on an object that has another object as its parent. See part 10 for more details.   
* **Tconnectedpart:** indicates a part/whole relationship where the part should be added to the scene in addition to the whole **and connected to it using an**  , e.g. the cap on a bottle. Is only valid on an object that has another object as its parent. See part 10 for more details.

### **NEW: Container Category / Synset Considerations**

Many original categories and synsets mapped *containers* of substances to the synsets corresponding to the substances themselves. This is problematic because in OmniGibson we have many substances and want to reserve the substance synsets for the actual substances.

As a result, for all containers (typically bottles / cans / etc.), we are creating custom synsets that reflect the fact that the container is a *container* of some *substance.* These synsets need to be named substance\_\_container.n.01, **(note: there are two underscores between the two parts)**, and the respective categories simply substance\_container (single underscore).

For example, a bottle of lemon juice can be put into a lemon\_juice\_bottle category which maps to a lemon\_juice\_\_bottle.n.01 synset. 

Typical options for the container part include bag, box, bottle, and can. For anything else, reach out to area leads for approval.

## 9\. Annotate lights

To annotate lights, you need to do two things: validate the properties and name the object correctly.

All lights in the scene need to be VrayLight objects. If you have PhysicalLight, TargetedLight, VrayIES, VrayAmbientLight, etc. objects (which SanityCheck will happily point out), ask for help.

For each such light object, the settings should look like this:  
![][image5.png]

That is:

* The type should be one of sphere, plane and disc. If this is not the case, ask for help.  
* The targeted option should be DISABLED. If it is enabled, click the checkbox to disable it, and if a corresponding VrayTarget object exists in the scene, remove it.  
* The units should be set to Luminous Power (lm). If something else is selected, switch this dropdown to this option. The multiplier should automatically get converted.  
* The multiplier should be a number roughly around the **500-1000** ballpark. Typically 800 for a regular 80W lightbulb. If this is not the case, ask for help.

Once the settings are all set, you need to name the object. Light sources need to be named such that they have the exact name of the object they will be attached to, plus a light identifier. For example, if the light source needs to be attached to *bottom\_cabinet-0-0* then it should be called *bottom\_cabinet-0-0-L0* where the 0 is the light index (and if there are multiple lights on this same object, that number should be incremented). If the light is attached to an articulated link, it should be named exactly as such: a light attached to *cabinet-0-0-drawer-base\_link-P-lower* will be called *cabinet-0-0-drawer-base\_link-P-lower-L0*.

Note that lights attached to moving parts should **only** be annotated for the lower-side copy of the moving part.

## 10\. NEW: Annotate Parts and Meta Links

Each object requires certain meta links to simulate its functionality in OmniGibson. For example, for a sink object to be functional, we need to annotate the water sources, drains, and toggle switches for each of its sink/drain combinations. Another example are objects that have parts: for example, a bottle might come with a separate cap object, or an apple object may be sliceable into parts. 

Before continuing into the details of meta links and parts, a quick note that meta links and parts need to be marked as **children** of the relevant object. To do this, you can drag the meta link or part in the Scene Explorer window and drop it on top of the related parent object.

A meta link needs to be a primitive object. Meta links can either be dimensionless (contains a pose but no volume definition) or volumetric (contains a pose and a volume definition) based on the requirements of the feature. First, you need to know if the annotation is going to be volumetric or dimensionless.

### **Dimensionless Meta Links**

Currently, annotations for the below meta links need to be dimensionless (e.g. **Helper \> Point** objects):

* **Mfluidsource:** the part(s) on the object where the relevant fluid is going to come out of  
* **Mattachment:** the point(s) on the object where an attachment will happen.  
  * You need to add a meta ID that matches the type in the list and a male/female option. For example, Mattachment\_displaywallM\_. **If the attachment type is labeled as pose in the list of attachment types, the attachment will only be created if the attachment meta links on the two objects fully line up (e.g. the orientation of the meta links is also important and will be taken into account).**  
    * **bicycle-abc-0-Mattachment\_bikerackM**  
    * **wardrobe-xyz-0-Mattachment\_hangerrackF\_0**  
* **Mheatsource:** the part(s) on the object where heat will be coming out  
  * Examples. stove, lighter  
  * Non-examples: microwave, stove, fridge (we only require objects to be inside these home appliances) 

To create one, you can go to Create \> Helpers \> Point.  
![][image6.png]

### **Volumetric Meta Links**

The below meta links need to be geometries, e.g. they need to contain volume. For these ones, you can use one of the **Box**, **Cone**, **Sphere** and **Cylinder** primitives.

* **Mtogglebutton:** the part(s) on the object where the robot can touch to toggle it on/off  
* **Mparticleapplier:** the volume through which particles will be applied. This might be something like a cone for objects that spray out particles.  
* **Mparticleremover:** the volume through which particles will be applied. This might be something like a box for objects like vacuums, where particles that enter the box will be removed.  
* **Mfluidsink:** the volume which, any particles of water (or the other relevant fluids) that enter will be removed / deleted  
* **Mslicer:** the volume such that if a sliceable object touches this object with the contact point falling within this volume, the sliceable object will be sliced.  
* **Mfillable:** the volume that will be checked for fluid particles when checking for the filled state.

To create one, you can go to the Create Menu, and the Standard Primitives submenu for the volumetric primitives:  
![][image7.png]

### **Meta IDs**

Some objects, such as a sink object that actually contains multiple sink/faucet pairs, might require multiple copies of a meta link. In the example of the sink, each faucet needs togglebutton and fluidsource meta links, and each drain needs a fluidsink meta link. Moreover, the meta links need to somehow be identified as a group such that the correct togglebutton will enable each nfluidsource.

To do this, we use *meta IDs*, an alphanumeric suffix attached to the meta link. For example, we can name one meta link *Mtogglebutton\_left* and one faucet *Mfluidsource\_left*. This will let OmniGibson know that these together are a pair of links that are connected. If a meta ID is not specified, it will default to “0”.

### **Meta Sub-IDs**

Some meta links on the other hand might not be best approximated with a single cylinder/cone/etc primitive. For example, many bottles have interior shapes that don’t match this shape. For these, we want to be able to annotate multiple primitives parts of the same meta-link. To do these, we simply name the parts of the meta link with both a meta ID (default 0\) and a meta sub ID (an integer, required consecutive, starting with zero). For example, two parts of a fillable meta link for a bottle can be called \-Mfillable\_0\_0 and \-Mfillable\_0\_1. **This feature is currently most useful for the fillable use case \- ask if you feel the need to use it for anything else.**

### **Primitive-specific considerations**

For now, the only primitive-specific consideration is that Cone primitives need to have their radius1 be zero and radius2 be nonzero. This means you should create the cone like a cylinder and then in the modify pane change the radii, because the default cone creation method only supports tapering radius2.

### **Parts**

For mechanisms like slicing, and also for importing “extra” objects that come with an object, we use parts. To annotate an object as a part of another, you can simply drag it under the parent object to make it its child, just like a meta link. If the part object is from another file, it can also be annotated as Bad. Part objects need to be Editable Poly. **They do not need to be named with any meta link information.**

There are three kinds of parts:

* Subparts are parts that should replace the main object in the event of slicing, etc. \- **that is, the parent object and the parts should never exist together in simulation**. In this case, the part needs to be tagged \-Tsubpart  
* Extra parts are parts that should be added into simulation along with the main object, despite being separate objects themselves. They will **not** be attached to the parent, e.g. they can move freely. For example, pillows on a bed, etc. \- these need to be tagged \-Textrapart  
* Connected parts are parts that should be added into simulation along with the main object, and **connected to the main object using an attachment** that can be broken with adequate force. For example, a cap on a bottle. These need to be tagged \-Tconnectedpart

## 11\. NEW: Assign rooms

In scene files, each object (including floors, ceilings, lights and cameras) need to be assigned into a room. Valid rooms types include: bar, bathroom, bedroom, biology\_lab, break\_room, chemistry\_lab, childs\_room, classroom, closet, computer\_lab, conference\_hall, copy\_room, corridor, dining\_room, empty\_room, entryway, exercise\_room, garage, garden, grocery\_store, gym, hammam, infirmary, kitchen, living\_room, lobby, locker\_room, meeting\_room, pantry\_room, phone\_room, playroom, private\_office, sauna, shared\_office, spa, staircase, storage\_room, television\_room, utility\_room. Each room should be named with a room type followed by an underscore and an index for that instance of that room type, e.g. living\_room\_0 or bathroom\_3.

To assign an object into a room, first create a layer with the name of the room in the layer manager, and drag the object onto that layer.

Some objects, such as doors and windows, might need to be assigned into multiple rooms. In those cases, you can create a layer whose name is a comma-separated list of rooms: e.g. corridor\_0,living\_room\_0

**Note that floors and ceilings are required to be per-room, e.g. there can be multiple floors/ceilings per room but never a floor or ceiling that extends to more than one room.**

The exceptions to the room assignment requirement are walls (which we will annotate later) and exterior cameras etc. that don’t exactly fit in a room. These can be left in the 0 layer.

## 12\. Verify you are passing the sanity check

Run the sanity check script to check that the object names are all valid, the instance setup is correct, and the categories are all whitelisted. The sanitycheck script has additional checks that it will perform and give you recommendations about. These need to be followed prior to being merged.

## 13\. Commit

After you are done with all of the scenes above, save the file one last time, commit the changes again if there are any remaining.

## 14\. Thank you\!

We are now one step closer to a very realistic household robotics simulator\!

# C. Object Acquisition Process

## 0\. Shop

The goal is to find objects of the task-relevant object categories on TurboSquid.

This is the [spreadsheet](https://docs.google.com/spreadsheets/d/1fWrP4DC4WSqgTEdRjXiIHO_ipmbT_HsEZ3CYvlgx0Wc/edit#gid=363727863) that keeps track of all the task-relevant objects. You can access it with your Stanford account.

* Column A: word (you should search it on TurboSquid or its synonym)  
* Column F: Eric comment (whether something is fluid, particle system, cloth, etc)  
* Column H: ​​Purchase link (you only need to go over the TRUE cell and copy/paste the TurboSquid purchase link to here)  
* Column I: is branded (you should put 1 there if the object is Editorial or Branded)

**Please also add the object into TurboSquid shopping cart.**

If Column F says Fluid or Particle, we should look for objects that are the natural/native container for these substances. Here are some examples:

Orange juice: [link](https://www.turbosquid.com/FullPreview/1637471)  
​​![][image8.png]

Flour: [link](https://www.turbosquid.com/3d-models/3d-wheat-flour-brown-paper-bag-2lb-model-1719748)  
![][image9.png]

A couple important notes (with decreasing importance):

* The quality should be high \- you can judge from the rendering result on their page  
* Native format has to be 3DS MAX (except Corona renderer; V-ray, scanline and arnold are okay).   
  * Note: selecting format as “.max (3DS MAX)” does NOT guarantee that the native format is 3DS MAX. We want it to be the case, so you would need to double check the “NATIVE” format on the product page.

  ![][image10.png]

  ![][image11.png]

* It should be affordable. Prioritize Free items. Our soft price cap is $20 per object. If you have to buy something more expensive, talk with us.   
  ![][image12.png]  
* **It should NOT be Editorial or Branded**. However, if you have to buy something editorial or branded, mark 1 in “Column I: is branded” of the spreadsheet. You will later edit the material in 3DS MAX to remove the brand.   
  ![][image13.png]

## 1\. Import files Into ig\_cad

Before we make any changes to the file, you need to import it into the ig\_cad repo. Follow the below steps to do that:

1. Only for your first file, do the following:  
   1. Check out the master branch (*git checkout master*).  
   2. Check the status of your current repo (*git status*) and check that there are no changes in your working copy. If there are, clear them.  
   3. Pull the latest master (*git pull*)  
   4. Create your objects branch. **All of your objects should be added to a single branch, and you should commit after each time you add a new object and each time you finish working on another object.** You should name your branch {yourname}-objects, e.g. cem-objects. To do this, run *git checkout \-b yourname-objects*.  
2. Make sure you are in your objects branch (*git checkout yourname-objects*).  
3. Download the next model from Turbosquid. To do this:  
   1. Click on the TurboSquid link in the [spreadsheet](https://docs.google.com/spreadsheets/d/1fWrP4DC4WSqgTEdRjXiIHO_ipmbT_HsEZ3CYvlgx0Wc/edit#gid=1576528389).  
   2. Add the object to the cart.  
   3. In the cart, see the download button. Click on that to start the download.  
   4. Remove the object from the cart (we don’t want to accidentally re-buy it later).  
   5. Save the downloaded file to a location that is not on the repo (e.g. your downloads folder). You can create an objects directory in your downloads folder to make this more manageable as we download hundreds of objects.  
   6. Extract your object’s downloaded files to a location that is **not on the repo.** You can create an objects directory in your downloads folder to make this more manageable as we download hundreds of objects.  
4. Open the file in 3ds Max. Make sure that the file loads fine without any texture or plugin errors. If you see any errors, let us know.  
5. In the right-hand menu, go into the Utilities tab (wrench icon). Open the Resource Collector utility, if not open, by going into More \> Resource Collector.  
6. In the resource collector, tick the *Collect Bitmaps*, *Copy* and *Update Materials* boxes. Then click the Browse button and navigate to the folder your raw.max file is in. Here, create a *textures* directory, and choose this directory. If these steps fail, instead go to file, click on archive, and select some directory. After archiving, unzip the file and in the resulting file navigate  all the way down to the folder containing assets. Copy these assets into the *textures* directory created earlier.  
7. Click Begin.  
8. Go into File \> Save As.  
   1. Go into the ig\_cad repository, into the *objects* directory, create a directory for your object file. This directory should be named {category}-{[random2charstring](https://www.random.org/strings/)}. **Note that this 2 char string is not meant to match the object model ID, it is only to prevent duplicate files here.** The category can come from the spreadsheet. If you have a file that contains multiple object categori es, instead of the category name, you can put in an arbitrary name that covers all of your objects names. For example: *raw\_meats-xy*.  
   2. Go into the directory you created.  
   3. Save the file as raw.max.  
9. Once resource collection finishes, save the file again by doing CTRL+S.  
10. **IMPORTANT:** time to commit\!  
    1. Go into ig\_cad, run *git status* and confirm that ONLY the folder you just added has changed.  
    2. Run *git add \-A*  
    3. **IMPORTANT:** run *git lfs status* and confirm that all of the files under textures/ show up here. (Cem: update on what to do here)  
    4. Run *git commit* *\-m your commit message* to commit the files.  
11. **IMPORTANT**: Go to the spreadsheet and put a 1 in the *“Raw File Added”* column on the spreadsheet.  
12. Then go into the File \> Save As and resave the file as processed.max (make sure you are saving in the correct folder.  
13. You are now ready to start working on the file\!

## 2\. Annotate the objects in the file

For any objects you see in the file, annotate them as usual using steps 2-10 from the scene guide above. **If there are multiple objects in the file you should still annotate them, as separate objects, and leave them in the same file.**

**IMPORTANT:** For the model ID component of the name (the first number after the category), use a [random 6-character string](https://www.random.org/strings/) rather than just an integer (0).

## 3\. Commit the File

After completing steps 2-10, from the scene guide and saving, run *git status*. You should see only the *processed.max* file added. Add, commit and push the file.

**IMPORTANT:** Go to the spreadsheet and put a 1 in the “Processing Completed” column on the spreadsheet.

## 4\. Repeat

Keep doing the above for all of the files assigned to you\!

# D. Scene Objects Category / Synset QA Process

For synset QA, the goal is to individually examine each object to make sure the category and synset assigned is correct. Instructions:

1. Open the next unprocessed scene in 3ds Max. The scene files are located in D:\\ig\_pipeline\\cad\\scenes\\{scene\_name}\\processed.max. You can start with the grocery store scenes.  
2. If prompted for anything (e.g. “scene converter”) hit X if possible. If prompted for units, ask to keep system units. **Do not save any changes onto the scene files.**  
3. Put the camera into Perspective mode by clicking on the first menu in top-left of 3d viewport, next to the \+, and picking Perspective (or just click on the viewport and press P).  
4. Make sure the Wireframe mode is disabled in the rightmost (3rd) menu next to the \+ on the top left of the viewport. Enable “Edged Faces” if you want to see the edges.  
5. Press F11 to open the scripting console to see messages.  
6. Repeatedly until you see the “Scene complete. Move to next scene.” message, do the below:  
   1. Click on the Object QA button on the top toolbar. The next object that is unprocessed will be selected. Hit Z to focus on the object.  
      1. **Important:** As soon as you click Object QA, the **next** object, which is opened and shown to you, will be marked **as completed.** As a result, make sure you have evaluated and finished work on the last object you see prior to finishing a QA session.  
   2. Read the category and synset assignment in the scripting console. Verify them per the “category/synset things to check for” list discussed below.  
   3. Take the following actions for any problems:  
      1. If the object can’t reasonably be mapped to anything (for example, we have a bowl of guacamole as a single rigid body), add the object to our [Deletion Queue](https://docs.google.com/spreadsheets/d/10L8wjNDvr1XYMMHas4IYYP9ZK7TfQHu--Kzoi0qhAe4/edit#gid=701868150) together with an explanation.  
      2. If the category name is correct, but the synset assignment is incorrect:  
         1. Decide on the appropriate synset, using the WordNet viewer if necessary.  
         2. Go to the [Category Mapping table](https://docs.google.com/spreadsheets/d/10L8wjNDvr1XYMMHas4IYYP9ZK7TfQHu--Kzoi0qhAe4/edit#gid=184473071), find the correct row and update the synset mapping. Remove the \`approved\` column value so that this updated entry will also be reviewed.  
      3. If the category name is incorrect:  
         1. Go to the [Object Rename table](https://docs.google.com/spreadsheets/d/10L8wjNDvr1XYMMHas4IYYP9ZK7TfQHu--Kzoi0qhAe4/edit#gid=1314512120). Enter the object name (category-6digitmodelID) into the first column. Enter the new category name into the third column. Verify that the “new synset” entry is what you expect.  
         2. If you need to update the synset too or add a synset for a new category, go to the [Category Mapping table](https://docs.google.com/spreadsheets/d/10L8wjNDvr1XYMMHas4IYYP9ZK7TfQHu--Kzoi0qhAe4/edit#gid=184473071), find the correct row if it exists and add/update the synset mapping. Remove the \`approved\` column value so that this updated entry will also be reviewed.  
7. Once you see the Scene complete message, mark the scene’s Object QA as completed on the [Scene Annotation table](https://docs.google.com/spreadsheets/d/10L8wjNDvr1XYMMHas4IYYP9ZK7TfQHu--Kzoi0qhAe4/edit#gid=0). Commit the QA pass file on Git to prevent data loss.  
8. Open the next scene file and repeat.

Category/synset things to check for:

* Per section B8: Many original categories and synsets mapped *containers* of substances to the synsets corresponding to the substances themselves. This is problematic because in OmniGibson we have many substances and want to reserve the substance synsets for the actual substances.  
  As a result, for all containers (typically bottles / cans / etc.), we are creating custom synsets that reflect the fact that the container is a *container* of some *substance.* These synsets need to be named substance\_\_container.n.01, **(note: there are two underscores between the two parts)**, and the respective categories simply substance\_container (single underscore).  
  For example, a bottle of lemon juice can be put into a lemon\_juice\_bottle category which maps to a lemon\_juice\_\_bottle.n.01 synset.   
  Typical options for the container part include bag, box, bottle, jar and can. For anything else, reach out to area leads for approval.

# E. Object QA Process

Setup process:

1. Open the GitHub Desktop app (make sure it's showing the ig\_pipeline branch)  
2. Check if it shows anything that's uncommitted. If it does, create a new branch and commit all the current changes. To do this, you click on the Current Branch button on the top menu, click New Branch, call it something like ruohan-temp, then tell it to Bring Over your work. Then on the left side select everything, and click the Commit button on the bottom left. The left side changed files list should be empty now.  
3. Then go back to the current branch menu up top and switch to main  
4. Click on the Fetch Origin button to the right of that (or if it shows Pull, skip to next step)  
5. Click on the Pull button (again to the right of the current branch menu)  
6. Download the dataset from [this link](https://storage.googleapis.com/gibson_scenes/dataset-5-3.zip) and extract it to D:\\dataset-5-3. Check that you extracted it correctly (e.g. D:\\dataset-5-3 should directly contain directories called objects, scenes, metadata)  
7. At this point you are ready to start the QA \- open a terminal window (“Windows Terminal”) and run the below:  
   1. conda activate pipeline  
   2. cd D:\\ig\_pipeline  
   3. python \-m b1k\_pipeline.qa\_viewer D:\\dataset-5-3 .\\qa-logs\\pass-one.json

This should launch the QA script with pybullet on the side (note that the pass-one record file lists some objects as already done which is intentional \- Mona did them).

QA process:

* The QA process will ask you a number of questions.  
*   For each question if you don't type in a complaint (e.g. you just hit enter) that means there is nothing wrong for that object for that question.  
* When you type something it gets recorded into a JSON file for that object which I will help you push later.   
* You will get six questions:  
  * the synset mapping  
  * whether the object looks OK (e.g. no glaring visual issues)  
  * whether it is roughly the right size (need to use some intuition for this)  
  * whether it MUST be simulated as a soft body e.g. we CAN’T realistically have this object be a rigid body. For example, blankets, clothes, etc. MUST be soft, whereas things like flowers can reasonably be rigid  
  * whether it has the right meta links (e.g. does it need a fillable volume? a water source? etc.)  
  * whether it has the right articulations and the articulations all look OK  
* For the first question (is synset OK) you should repeat the process from before, e.g. if the synset assignment is not OK then you can simply add the entry into the rename table \- you shouldn't put in a complaint on the QA system.  
* For the fifth question:  
  * fluidsource: whether this object should have a fluid source location (typically for infinite stuff like sinks, deodorants, etc.)  
  * togglebutton: a toggle button for enabling heating/fluid functionality  
  * heatsource: if this object should have a location-based heat source (e.g. like a stove, lighter, candle). not needed for stuff like microwaves and ovens where it suffices to be inside  
  * particleapplier: whether this object should have a particle spraying location  
  * particleremover: whether this object should have a particle removing location (useful for cleaning tools, vacuums, lawnmower, etc)  
  * fluidsink: whether this object should have a region that removes particles on contact, like a sink drain  
  * slicer: whether this object should have a region that slices objects upon contact  
  * fillable: whether this object should have a volume that can be checked for "filled" \- we should be able to generate this requirement from the BDDLs and it's not possible for you to tell whether the object has a reachable inside so it's OK to skip  
  * attachment: should this object have any attachment points (no need to do this, I have a list of stuff that do)  
* After you answer the 6th question it will move onto the next object. Comments are recorded every time you hit enter, objects are marked as done when you finish all questions about an object.  
* You can CTRL+C out any time and continue where you left off by running the same python  command.  
* I simply used low-res v-hacd for the collision meshes currently so the objects may not have correct collision meshes \- that is not a problem.

