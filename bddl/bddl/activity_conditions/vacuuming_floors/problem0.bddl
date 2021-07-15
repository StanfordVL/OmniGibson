(define (problem vacuuming_floors_0)
    (:domain igibson)

    (:objects
     	floor.n.01_1 - floor.n.01
    	vacuum.n.04_1 - vacuum.n.04
        ashcan.n.01_1 - ashcan.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (dusty floor.n.01_1) 
        (onfloor vacuum.n.04_1 floor.n.01_1) 
        (onfloor ashcan.n.01_1 floor.n.01_1) 
        (inroom floor.n.01_1 bedroom) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (not 
                (dusty ?floor.n.01_1)
            )
        )
    )
)