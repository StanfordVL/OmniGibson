(define (problem brushing_lint_off_clothing_0)
    (:domain igibson)

    (:objects
     	sweater.n.01_1 sweater.n.01_2 sweater.n.01_3 sweater.n.01_4 - sweater.n.01
    	floor.n.01_1 - floor.n.01
    	bed.n.01_1 - bed.n.01
    	scrub_brush.n.01_1 - scrub_brush.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor sweater.n.01_1 floor.n.01_1) 
        (onfloor sweater.n.01_2 floor.n.01_1) 
        (ontop sweater.n.01_3 bed.n.01_1) 
        (ontop sweater.n.01_4 bed.n.01_1) 
        (dusty sweater.n.01_1) 
        (dusty sweater.n.01_2) 
        (dusty sweater.n.01_3) 
        (dusty sweater.n.01_4) 
        (onfloor scrub_brush.n.01_1 floor.n.01_1) 
        (not 
            (dusty scrub_brush.n.01_1)
        ) 
        (inroom floor.n.01_1 bedroom) 
        (inroom bed.n.01_1 bedroom) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forall 
                (?sweater.n.01 - sweater.n.01) 
                (not 
                    (dusty ?sweater.n.01)
                )
            ) 
            (forall 
                (?sweater.n.01 - sweater.n.01) 
                (ontop ?sweater.n.01 ?bed.n.01_1)
            )
        )
    )
)