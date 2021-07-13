(define (problem preserving_food_0)
    (:domain igibson)

    (:objects 
        fish.n.02_1 - fish.n.02
        spatula.n.01_1 - spatula.n.01
        saucepan.n.01_1 - saucepan.n.01
        shelf.n.01_1 - shelf.n.01
        stove.n.01_1 - stove.n.01
        floor.n.01_1 - floor.n.01
        agent.n.01_1 - agent.n.01
    )

    (:init 
        (ontop fish.n.02_1 shelf.n.01_1)
        (ontop spatula.n.01_1 shelf.n.01_1)
        (ontop saucepan.n.01_1 shelf.n.01_1)
        (inroom shelf.n.01_1 kitchen)
        (inroom stove.n.01_1 kitchen)
        (inroom floor.n.01_1 kitchen)
        (onfloor agent.n.01_1 floor.n.01_1)
        (toggled_on stove.n.01_1) 
    )

    (:goal 
        (and 
            (cooked fish.n.02_1)
        )
    )
)