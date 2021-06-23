(define (problem loading_the_dishwasher_0)
    (:domain igibson)

    (:objects
        plate.n.04_1 plate.n.04_2 plate.n.04_3 - plate.n.04
       countertop.n.01_1 - countertop.n.01
        mug.n.04_1 - mug.n.04
        bowl.n.01_1 bowl.n.01_2 - bowl.n.01
        dishwasher.n.01_1 - dishwasher.n.01
        floor.n.01_1 - floor.n.01
        agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (ontop plate.n.04_1 countertop.n.01_1) 
        (stained plate.n.04_1) 
        (ontop plate.n.04_2 countertop.n.01_1) 
        (stained plate.n.04_2) 
        (ontop plate.n.04_3 countertop.n.01_1) 
        (stained plate.n.04_3) 
        (ontop mug.n.04_1 countertop.n.01_1) 
        (stained mug.n.04_1) 
        (ontop bowl.n.01_1 countertop.n.01_1) 
        (stained bowl.n.01_1) 
        (ontop bowl.n.01_2 countertop.n.01_1) 
        (stained bowl.n.01_2) 
        (inroom dishwasher.n.01_1 kitchen) 
        (inroom countertop.n.01_1 kitchen) 
        (inroom floor.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forall 
                (?plate.n.04 - plate.n.04) 
                (inside ?plate.n.04 ?dishwasher.n.01_1)
            ) 
            (forall 
                (?bowl.n.01 - bowl.n.01) 
                (inside ?bowl.n.01 ?dishwasher.n.01_1)
            ) 
            (inside ?mug.n.04_1 ?dishwasher.n.01_1)
        )
    )
)
