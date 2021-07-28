# Behavior Domain Definition Language 

The Behavior Domain Definition Language (BDDL) is a domain-specific language designed for the Benchmark for Everyday Household Activities in Virtual, Interactive, and ecOlogical enviRonments (BEHAVIOR). 

BDDL is a predicate logic-based language inspired by, but distinct from, the Planning Domain Definition Language [1]. It defines each BEHAVIOR activity definition as a BDDL `problem`, consisting of of a categorized object list (`:objects`), an initial condition that has only ground literals (`:init`), and a goal condition that is a logical expression (`:goal`). 

## Installation

To install this implementation of BDDL, clone this repository locally:
```
git clone https://github.com/StanfordVL/bddl.git
```
then run setup: 
```
cd bddl
python setup.py install
```

## Example BDDL activity

```
(define 
    (problem cleaning_the_pool_0)
    (:domain igibson)

    (:objects
     	pool.n.01_1 - pool.n.01
    	floor.n.01_1 - floor.n.01
    	scrub_brush.n.01_1 - scrub_brush.n.01
    	shelf.n.01_1 - shelf.n.01
    	detergent.n.02_1 - detergent.n.02
        sink.n.01_1 - sink.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor pool.n.01_1 floor.n.01_1) 
        (stained pool.n.01_1) 
        (onfloor scrub_brush.n.01_1 floor.n.01_1) 
        (onfloor detergent.n.02_1 floor.n.01_1) 
        (inroom shelf.n.01_1 garage) 
        (inroom floor.n.01_1 garage) 
        (inroom sink.n.01_1 storage_room)
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (onfloor ?pool.n.01_1 ?floor.n.01_1) 
            (not 
                (stained ?pool.n.01_1)
            ) 
            (ontop ?scrub_brush.n.01_1 ?shelf.n.01_1) 
            (onfloor ?detergent.n.02_1 ?floor.n.01_1)
        )
    )
)
```

The `:objects` and `:init` sections specify the initial state than an agent will start in, located as specified in `:init`. The `inroom` predicate specifies which scene objects must be present, and other binary kinematic predicates (`ontop`, `inside`, etc.) specify where small objects should be sampled. The BDDL functionality sends a representation of these conditions to sampling functionality implemented in a simulator (such as iGibson 2.0) to be sampled into a physical instance of the activity. 

The `:goal` section specifies the condition that the agent must satisfy to be successful on the activity. BDDL is entirely process-agnostic, specifiying only the simulator state that must be reached for success. 

## Example code usage

### Without simulator 

You will typically want to use BEHAVIOR activities with a simulator. To use a BEHAVIOR activity without a simulator, use the following code. 
```
from bddl.activity_base import BEHAVIORActivityInstance 
behavior_activity = "storing_the_groceries"     # the activity you want to try, full list in bddl/bddl/activity_conditions
activity_definition = 0                         # the specific definition you want to use. As of BEHAVIOR100 2021, this should always be 0.

behavior_activity_instance = BEHAVIORActivityInstance(behavior_activity=behavior_activity, activity_definition=activity_definition)
```

### With simulator 

To use a BEHAVIOR activity with a simulator, create a subclass of `BEHAVIORActivityInstance` for your simulator. Example for iGibson 2.0. This will require an implementation of sampling functionality or pre-sampled scenes that satisfy the activity's initial condition and implementation for checking each type of binary kinematic predicate (e.g. `ontop`, `nextto`) and unary nonkinematic predicate (e.g. `cooked`, `soaked`). 

## Logic evaluator for goal

When using BEHAVIOR activities with a simulator, the goal condition is evaluated at every simulator step by calling `simulator_activity_instance.check_success()`, where `simulator_activity_instance` is some subclass of `BEHAVIORActivityInstance`. `bddl.logic_base` and `bddl.condition_evaluation` contain this functionality. Atomic formulae that interface directly with the simulator are implemented in `bddl.logic_base`. These require the simulator checking functions for various predicates to be implemented, and are the leaf nodes of the compositional expression making up a goal condition or the list of literals making up an initial condition. Logical operators are implemented in `bddl.condition_evaluation`, and form a compositional structure of the condition to evaluate. 

## Solver for ground goal solutions

`bddl.condition_evaluation` also contains basic functionality to generate ground solutions to a compositional goal condition, including one that may contain quantification. This functionality is much like a very simple, unoptimized logic program, and will return a subset of solutions in cases where the solution set is too large to compute due to exponential growth. 

# Using BEHAVIOR with a new simulator 

Using BEHAVIOR activities with a new simulator requires implementing its functional requirements for that simulator, as has been done for iGibson 2.0 [3]. 

### Implementation of BDDL predicates as simulated object states

To simulate a BEHAVIOR activity, the simulator must be able to simulate every predicate involved in that activity. The full list of predicates is at [TODO add list of predicates to config]. For any one activity, the required predicates can be found by reading its BDDL problem (in activity_conditions/<activity_name>/.) 

Implementing these requires 1) a simulator-specific child class of the `BDDLBackend` class ([link](https://github.com/StanfordVL/bddl/blob/654cfefb078dbdf264957a08a30571086a2aa726/bddl/backend_abc.py#L6-L9)) and 2) implementations of object states such as `cooked` and `ontop` that can both **instantiate** an object as e.g. `cooked` or `not cooked`, and **check** whether the predicate is true for a given object. 

**1. Child of `BDDLBACKEND`:** This class has one method, `get_predicate_class`. It must take string tokens of predicates from BDDL problems (e.g. `"cooked"`, `"ontop"`) and map them to the simulator's object states. Example: [iGibson's `BDDLBackend` child class](https://github.com/StanfordVL/iGibson/blob/ig-develop/igibson/task/bddl_backend.py). 

**2. Simulated object states:** For any object in a BEHAVIOR activity, it must be instantiated in certain simulated states and be checked for certain simulated states, as specified by a BDDL problem. `BDDLBackend` expects state implementations that are object agnostic, but the implementation is ultimately up to the user. Assuming object-agnostic states, each one should be able to take an object and instantiate that object with the given state if applicable, and check whether that object is in that state or not. Example: [iGibson's object state implementations](https://github.com/StanfordVL/iGibson/tree/ig-develop/igibson/object_states). 

*Note on binary predicates:* in BDDL, all binary predicates are kinematic (`ontop`, `nextto`, `touching`, etc.). Instantiating objects in the associated simulator states is more complex than instantiating objects in unary predicates' states due to potential for failure based on physical constraints of the scene and multiple possibilities for object pairing, especially when implementing scene-agnostic instantiation capable of generating infinite distinct episodes. Please look at the setter methods of kinematic states in iGibson 2.0 for a robust example capable of instantiating BEHAVIOR activities with many objects. 

# Testing

To test the predicate evaluator, run `pytest` in project root.

To add a test, create a new python file under the tests directory, and add
additional functions prefixed with `test_` which include assert statements that
should evaluate true.

# References 

[1] https://www.researchgate.net/publication/2278933_PDDL_-_The_Planning_Domain_Definition_Language#:~:text=This%20planning%20domain%20consists%20of,domain%20automatically%20from%20human%20demonstrations.

[2] TODO parsing code citation

[3] iG2.0