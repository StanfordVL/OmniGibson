(define (problem cleaning_cupboards_0)
    (:domain igibson)

    (:objects
     	cabinet.n.01_1 cabinet.n.01_2 - cabinet.n.01
    	rag.n.01_1 - rag.n.01
    	floor.n.01_1 - floor.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (dusty cabinet.n.01_1) 
        (dusty cabinet.n.01_2) 
        (onfloor rag.n.01_1 floor.n.01_1) 
        (not 
            (soaked rag.n.01_1)
        ) 
        (inroom cabinet.n.01_1 bathroom) 
        (inroom cabinet.n.01_2 closet) 
        (inroom floor.n.01_1 bathroom) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (soaked ?rag.n.01_1) 
            (not 
                (dusty ?cabinet.n.01_1)
            ) 
            (not 
                (dusty ?cabinet.n.01_2)
            )
        )
    )
)