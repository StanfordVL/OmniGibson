(define (problem preserving_food_0)
    (:domain igibson)

    (:objects 
        dishtowel.n.01_1 dishtowel.n.01_2 - dishtowel.n.01
        mug.n.04_1 - mug.n.04
        sink.n.01_1 - sink.n.01
        shelf.n.01_1 - shelf.n.01
        stove.n.01_1 - stove.n.01
        dishwasher.n.01_1 - dishwasher.n.01
        floor.n.01_1 - floor.n.01
        agent.n.01_1 - agent.n.01
    )

    (:init 
        (ontop dishtowel.n.01_1 dishwasher.n.01_1)
        (ontop dishtowel.n.01_2 dishwasher.n.01_1)
        (ontop mug.n.04_1 dishwasher.n.01_1)
        (inroom sink.n.01_1 kitchen)
        (inroom shelf.n.01_1 kitchen)
        (inroom stove.n.01_1 kitchen)
        (inroom dishwasher.n.01_1 kitchen)
        (inroom floor.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
        (stained shelf.n.01_1)
        (dusty stove.n.01_1)
    )

    (:goal 
        (and 
            (soaked dishtowel.n.01_1)
            (soaked dishtowel.n.01_2)
            (not (stained shelf.n.01_1))
            (not (dusty stove.n.01_1))
        )
    )
)