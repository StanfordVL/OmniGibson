(define (problem cleaning_freezer_1)
    (:domain igibson)

    (:objects
     	electric_refrigerator.n.01_1 - electric_refrigerator.n.01
    	towel.n.01_1 - towel.n.01
    	countertop.n.01_1 - countertop.n.01
    	floor.n.01_1 - floor.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (stained electric_refrigerator.n.01_1) 
        (ontop towel.n.01_1 countertop.n.01_1) 
        (inroom electric_refrigerator.n.01_1 kitchen) 
        (inroom countertop.n.01_1 kitchen) 
        (inroom floor.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (not 
                (stained ?electric_refrigerator.n.01_1)
            )
        )
    )
)