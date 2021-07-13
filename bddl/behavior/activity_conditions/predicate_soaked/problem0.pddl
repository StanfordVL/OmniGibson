(define (problem preserving_food_0)
    (:domain igibson)

    (:objects 
        towel.n.01_1 - towel.n.01
        sink.n.01_1 - sink.n.01
        dishwasher.n.01_1 - dishwasher.n.01
        floor.n.01_1 - floor.n.01
        agent.n.01_1 - agent.n.01
    )

    (:init 
        (ontop towel.n.01_1 dishwasher.n.01_1)
        (inroom sink.n.01_1 kitchen) 
        (inroom dishwasher.n.01_1 kitchen)
        (inroom floor.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
        (toggled_on sink.n.01_1) 
    )

    (:goal 
        (and 
            (soaked towel.n.01_1)
        )
    )
)