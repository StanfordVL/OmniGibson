(define (problem preserving_food_0)
    (:domain igibson)

    (:objects 
        towel.n.01_1 - towel.n.01
        shelf.n.01_1 - shelf.n.01
        floor.n.01_1 - floor.n.01
        agent.n.01_1 - agent.n.01
    )

    (:init 
        (ontop towel.n.01_1 shelf.n.01_1)
        (inroom shelf.n.01_1 kitchen)
        (inroom floor.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
        (soaked towel.n.01_1)
        (stained shelf.n.01_1)
    )

    (:goal 
        (and 
            (not (stained shelf.n.01_1))
        )
    )
)