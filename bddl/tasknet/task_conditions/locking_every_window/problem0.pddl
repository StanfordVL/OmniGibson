(define (problem locking_every_window_0)
    (:domain igibson)

    (:objects
     	window.n.01_1 window.n.01_2 window.n.01_3 window.n.01_4 - window.n.01
    	bed.n.01_1 - bed.n.01
    	sofa.n.01_1 - sofa.n.01
    	floor.n.01_1 - floor.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (open window.n.01_1) 
        (open window.n.01_2) 
        (open window.n.01_3) 
        (open window.n.01_4) 
        (inroom window.n.01_1 bedroom) 
        (inroom window.n.01_2 kitchen) 
        (inroom window.n.01_3 living_room) 
        (inroom window.n.01_4 living_room) 
        (inroom bed.n.01_1 bedroom) 
        (inroom sofa.n.01_1 living_room) 
        (inroom floor.n.01_1 living_room) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (not 
                (open ?window.n.01_1)
            ) 
            (not 
                (open ?window.n.01_2)
            ) 
            (not 
                (open ?window.n.01_3)
            ) 
            (not 
                (open ?window.n.01_4)
            )
        )
    )
)