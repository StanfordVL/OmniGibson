(define (problem cleaning_floors_0)
    (:domain igibson)

    (:objects
     	floor.n.01_1 - floor.n.01
    	broom.n.01_1 - broom.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (dusty floor.n.01_1) 
        (onfloor broom.n.01_1 floor.n.01_1) 
        (inroom floor.n.01_1 bathroom) 
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