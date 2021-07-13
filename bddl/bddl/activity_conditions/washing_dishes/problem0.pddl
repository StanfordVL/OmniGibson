(define (problem washing_dishes_0)
    (:domain igibson)

    (:objects
     	plate.n.04_1 plate.n.04_2 - plate.n.04
        cup.n.01_1 - cup.n.01
    	sink.n.01_1 - sink.n.01
    	bowl.n.01_1 - bowl.n.01
    	floor.n.01_1 - floor.n.01
        countertop.n.01_1 - countertop.n.01
        scrub_brush.n.01_1 - scrub_brush.n.01
        agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (ontop plate.n.04_1 countertop.n.01_1) 
        (ontop plate.n.04_2 countertop.n.01_1) 
        (ontop bowl.n.01_1 countertop.n.01_1) 
        (ontop cup.n.01_1 countertop.n.01_1) 
        (inside scrub_brush.n.01_1 sink.n.01_1) 
        (stained plate.n.04_1) 
        (stained plate.n.04_2) 
        (stained cup.n.01_1) 
        (stained bowl.n.01_1) 
        (inroom sink.n.01_1 kitchen) 
        (inroom countertop.n.01_1 kitchen) 
        (inroom floor.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forall 
                (?plate.n.04 - plate.n.04) 
                (not 
                    (stained ?plate.n.04)
                )
            ) 
            (not 
                (stained ?cup.n.01_1)
            ) 
            (not 
                (stained ?bowl.n.01_1)
            )
        )
    )
)