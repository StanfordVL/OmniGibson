(define (problem locking_every_door_0)
    (:domain igibson)

    (:objects
     	door.n.01_1 door.n.01_2 - door.n.01
    	bed.n.01_1 - bed.n.01
    	floor.n.01_1 - floor.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (open door.n.01_1) 
        (open door.n.01_2) 
        (inroom door.n.01_1 bedroom) 
        (inroom door.n.01_2 bathroom) 
        (inroom bed.n.01_1 bedroom) 
        (inroom floor.n.01_1 bedroom) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (not 
                (open ?door.n.01_1)
            ) 
            (not 
                (open ?door.n.01_2)
            )
        )
    )
)
