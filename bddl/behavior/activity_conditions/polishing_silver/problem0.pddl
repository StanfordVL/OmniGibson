(define (problem polishing_silver_0)
    (:domain igibson)

    (:objects
     	spoon.n.01_1 spoon.n.01_2 spoon.n.01_3 spoon.n.01_4 - spoon.n.01
    	rag.n.01_1 - rag.n.01
    	cabinet.n.01_1 - cabinet.n.01
    	floor.n.01_1 - floor.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (dusty spoon.n.01_1) 
        (dusty spoon.n.01_2) 
        (dusty spoon.n.01_3) 
        (dusty spoon.n.01_4) 
        (inside rag.n.01_1 cabinet.n.01_1) 
        (inside spoon.n.01_1 cabinet.n.01_1) 
        (inside spoon.n.01_2 cabinet.n.01_1) 
        (inside spoon.n.01_3 cabinet.n.01_1) 
        (inside spoon.n.01_4 cabinet.n.01_1) 
        (inroom cabinet.n.01_1 kitchen) 
        (inroom floor.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (not 
                (dusty ?spoon.n.01_1)
            ) 
            (not 
                (dusty ?spoon.n.01_2)
            ) 
            (not 
                (dusty ?spoon.n.01_3)
            ) 
            (not 
                (dusty ?spoon.n.01_4)
            ) 
            (not 
                (inside ?rag.n.01_1 ?cabinet.n.01_1)
            ) 
            (inside ?spoon.n.01_1 ?cabinet.n.01_1) 
            (inside ?spoon.n.01_2 ?cabinet.n.01_1) 
            (inside ?spoon.n.01_3 ?cabinet.n.01_1) 
            (inside ?spoon.n.01_4 ?cabinet.n.01_1)
        )
    )
)