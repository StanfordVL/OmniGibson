---
icon: material/book-plus
---

# **Related Research**

## [BEHAVIOR Robot Suite](https://behavior-robot-suite.github.io/)

BEHAVIOR Robot Suite (BRS) is a comprehensive framework for whole-body manipulation in household tasks, addressing key robotics challenges including bimanual coordination, stable navigation, and extensive reachability. Built on a bimanual wheeled robot with a 4-DoF torso, BRS integrates a cost-effective teleoperation interface for data collection and a novel algorithm for learning whole-body visuomotor policies. The system is evaluated on five challenging household tasks involving long-range navigation, articulated and deformable object interaction, and manipulation in confined spaces, representing a significant step toward real-world household robotics.

## [Automated Creation of Digital Cousins for Robust Policy Learning](https://digital-cousins.github.io/)

ACDC (Automatic Creation of Digital Cousins) is a method that addresses the limitations of real-world robot training and simulation-to-real transfer by introducing "digital cousins" - virtual environments that capture similar geometric and semantic affordances to real scenes without explicitly modeling them. Unlike expensive digital twins, digital cousins provide cost-effective virtual replicas while enabling better cross-domain generalization through diverse similar training scenes. ACDC offers a fully automated real-to-sim-to-real pipeline that generates interactive scenes and trains robot policies for zero-shot deployment, achieving 90% success rates compared to 25% for digital twin-trained policies.

## [BEHAVIOR Vision Suite](https://behavior-vision-suite.github.io/)

BEHAVIOR Vision Suite is a synthetic data generator that addresses limitations in current computer vision datasets by providing high-quality, controllable data generation. It offers adjustable parameters at scene, object, and camera levels, enabling researchers to conduct controlled experiments by systematically varying conditions during data generation. The system supports three key applications: evaluating model robustness across domain shifts, testing scene understanding models on identical datasets, and training simulation-to-real transfer for state prediction tasks.

## [BEHAVIOR-1K](https://behavior.stanford.edu/behavior-1k)

BEHAVIOR-1K is a comprehensive simulation benchmark for human-centered robotics. Compared to its predecessor, BEHAVIOR-100, this new benchmark is more grounded on actual human needs: the 1,000 activities come from the results of an extensive survey on “what do you want robots to do for you?”. It is more diverse in the type of scenes, objects, and activities. Powered by NVIDIA’s Omniverse, BEHAVIOR-1K also achieves a new level of realism in rendering and physics simulation. We hope that BEHAVIOR-1K’s human-grounded nature, diversity, and realism make it valuable for embodied AI and robot learning research.

## [BEHAVIOR-100](https://behavior.stanford.edu/behavior-100)

BEHAVIOR-100 is the first generation of BEHAVIOR, a benchmark for embodied AI with 100 activities in simulation, spanning a range of everyday household chores such as cleaning, maintenance, and food preparation. These activities are designed to be realistic, diverse and complex, aiming to reproduce the challenges that agents must face in the real world.

## Gibson Series

### [Gibson](http://gibsonenv.stanford.edu/)

Gibson Environment is a virtual environment designed to address the challenges of developing real-world perception for active agents, where physical training is slow, costly, and impractical. Built on virtualized real spaces rather than artificial designs, Gibson includes over 1400 floor spaces from 572 buildings. Its key features include real-world semantic complexity, an internal synthesis mechanism called "Goggles" that enables direct real-world deployment without domain adaptation, and physics-based agent embodiment that reflects real-world spatial constraints.

### [iGibson 1.0](https://svl.stanford.edu/igibson/)

iGibson 1.0 is a simulation environment for developing robotic solutions for interactive tasks in realistic large-scale scenes. It features 15 fully interactive home-sized scenes with 108 rooms containing rigid and articulated objects that replicate real-world homes. Key capabilities include high-quality virtual sensor signals, domain randomization for object materials and shapes, integrated motion planners for collision-free trajectories, and an intuitive human interface for collecting demonstrations. The environment enables agents to learn useful visual representations, supports navigation generalization, and facilitates efficient imitation learning of human-demonstrated manipulation behaviors.

### [iGibson 2.0](https://svl.stanford.edu/igibson/)

iGibson 2.0 is an open-source simulation environment that expands beyond motion and physical contact to support diverse household tasks through three key innovations. It simulates object states (temperature, wetness, cleanliness, toggled/sliced states), implements predicate logic functions that map simulator states to logic states and can generate infinite task instances, and includes a VR interface for collecting human demonstrations. These capabilities enable more densely populated scenes with semantically meaningful object placement and support robot learning of novel tasks through imitation learning from human demonstrations.

