(define (problem polishing_shoes_0)
    (:domain igibson)

    (:objects
    	shoe.n.01_1 shoe.n.01_2 - shoe.n.01
    	rag.n.01_1 - rag.n.01
    	floor.n.01_1 - floor.n.01
    	sink.n.01_1 - sink.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (stained shoe.n.01_1) 
        (stained shoe.n.01_2) 
        (onfloor rag.n.01_1 floor.n.01_1) 
        (not 
            (soaked rag.n.01_1)
        ) 
        (onfloor shoe.n.01_1 floor.n.01_1) 
        (onfloor shoe.n.01_2 floor.n.01_1) 
        (inroom sink.n.01_1 bathroom) 
        (inroom floor.n.01_1 bathroom) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (nextto ?rag.n.01_1 ?sink.n.01_1) 
            (soaked ?rag.n.01_1) 
            (and 
                (not 
                    (stained ?shoe.n.01_1)
                ) 
                (not 
                    (stained ?shoe.n.01_2)
                )
            )
        )
    )
)