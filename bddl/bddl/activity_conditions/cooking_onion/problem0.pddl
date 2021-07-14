(define (problem preserving_food_0)
    (:domain igibson)

    (:objects 
        vidalia_onion.n.01_1 - vidalia_onion.n.01
        spatula.n.01_1 - spatula.n.01
        saucepan.n.01_1 - saucepan.n.01
        shelf.n.01_1 - shelf.n.01
        stove.n.01_1 - stove.n.01
        electric_refrigerator.n.01_1 - electric_refrigerator.n.01
        carving_knife.n.01_1 - carving_knife.n.01
        chopping_board.n.01_1 - chopping_board.n.01
        dishwasher.n.01_1 - dishwasher.n.01
        floor.n.01_1 - floor.n.01
        agent.n.01_1 - agent.n.01
    )

    (:init 
        (inside vidalia_onion.n.01_1 electric_refrigerator.n.01_1)
        (ontop spatula.n.01_1 shelf.n.01_1)
        (ontop saucepan.n.01_1 shelf.n.01_1)
        (ontop carving_knife.n.01_1 shelf.n.01_1)
        (ontop chopping_board.n.01_1 dishwasher.n.01_1)
        (inroom shelf.n.01_1 kitchen)
        (inroom stove.n.01_1 kitchen)
        (inroom floor.n.01_1 kitchen)
        (inroom electric_refrigerator.n.01_1 kitchen)
        (inroom dishwasher.n.01_1 kitchen)
        (onfloor agent.n.01_1 floor.n.01_1)
    )

    (:goal 
        (and 
            (burnt vidalia_onion.n.01_1)
            (sliced vidalia_onion.n.01_1)
        )
    )
)