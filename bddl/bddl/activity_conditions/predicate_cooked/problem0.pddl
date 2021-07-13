(define (problem preserving_food_0)
    (:domain igibson)

    (:objects 
        meat.n.01_1 - meat.n.01
        shelf.n.01_1 - shelf.n.01
        stove.n.01_1 - stove.n.01
        floor.n.01_1 - floor.n.01
        agent.n.01_1 - agent.n.01
    )

    (:init 
        (ontop meat.n.01_1 shelf.n.01_1)
        (inroom shelf.n.01_1 kitchen)
        (inroom stove.n.01_1 kitchen)
        (inroom floor.n.01_1 kitchen)
        (onfloor agent.n.01_1 floor.n.01_1)
        (toggled_on stove.n.01_1) 
    )

    (:goal 
        (and 
            (cooked meat.n.01_1)
        )
    )
)