(define (problem cleaning_carpets_1)
    (:domain igibson)

    (:objects
     	floor.n.01_1 - floor.n.01
    	shoe.n.01_1 - shoe.n.01
    	ball.n.01_1 - ball.n.01
    	vacuum.n.04_1 - vacuum.n.04
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (dusty floor.n.01_1) 
        (onfloor shoe.n.01_1 floor.n.01_1) 
        (onfloor ball.n.01_1 floor.n.01_1) 
        (onfloor vacuum.n.04_1 floor.n.01_1) 
        (inroom floor.n.01_1 corridor) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (not 
                (dusty ?floor.n.01_1)
            ) 
            (not 
                (onfloor ?shoe.n.01_1 ?floor.n.01_1)
            ) 
            (not 
                (onfloor ?ball.n.01_1 ?floor.n.01_1)
            )
        )
    )
)