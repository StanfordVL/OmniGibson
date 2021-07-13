(define (problem cleaning_stove_0)
    (:domain igibson)

    (:objects
     	stove.n.01_1 - stove.n.01
    	soap.n.01_1 - soap.n.01
    	cabinet.n.01_1 - cabinet.n.01
    	rag.n.01_1 - rag.n.01
    	sink.n.01_1 - sink.n.01
    	dishtowel.n.01_1 - dishtowel.n.01
            floor.n.01_1 - floor.n.01
            agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (dusty stove.n.01_1) 
        (stained stove.n.01_1) 
        (inside soap.n.01_1 cabinet.n.01_1) 
        (inside rag.n.01_1 cabinet.n.01_1) 
        (not 
            (soaked rag.n.01_1)
        ) 
        (not 
            (stained rag.n.01_1)
        ) 
        (inside dishtowel.n.01_1 cabinet.n.01_1) 
        (not 
            (soaked dishtowel.n.01_1)
        ) 
        (inroom sink.n.01_1 kitchen) 
        (inroom stove.n.01_1 kitchen) 
        (inroom cabinet.n.01_1 kitchen) 
        (inroom floor.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (not 
                (dusty ?stove.n.01_1)
            ) 
            (not 
                (stained ?stove.n.01_1)
            ) 
            (nextto ?rag.n.01_1 ?sink.n.01_1) 
            (nextto ?dishtowel.n.01_1 ?sink.n.01_1)
        )
    )
)