(define (problem cleaning_carpets_1)
    (:domain igibson)

    (:objects
     	floor.n.01_1 floor.n.01_2 - floor.n.01
    	vacuum.n.04_1 - vacuum.n.04
    	washer.n.03_1 - washer.n.03
    	dryer.n.01_1 - dryer.n.01
        agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (dusty floor.n.01_1) 
        (dusty floor.n.01_2) 
        (onfloor vacuum.n.04_1 floor.n.01_1) 
        (inroom washer.n.03_1 utility_room) 
        (inroom dryer.n.01_1 utility_room) 
        (inroom floor.n.01_1 corridor) 
        (inroom floor.n.01_2 utility_room) 
        (onfloor agent.n.01_1 floor.n.01_2)
    )
    
    (:goal 
        (and 
            (forall 
                (?floor.n.01 - floor.n.01) 
                (not 
                    (dusty ?floor.n.01)
                )
            ) 
        )
    )
)