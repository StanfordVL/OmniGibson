(define (problem polishing_furniture_0)
    (:domain igibson)

    (:objects
     	shelf.n.01_1 - shelf.n.01
    	table.n.02_1 - table.n.02
    	rag.n.01_1 - rag.n.01
    	floor.n.01_1 - floor.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (dusty shelf.n.01_1) 
        (dusty table.n.02_1) 
        (ontop rag.n.01_1 table.n.02_1) 
        (inroom table.n.02_1 living_room) 
        (inroom shelf.n.01_1 living_room) 
        (inroom floor.n.01_1 living_room) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (not 
                (dusty ?table.n.02_1)
            ) 
            (not 
                (dusty ?shelf.n.01_1)
            ) 
            (under ?rag.n.01_1 ?table.n.02_1)
        )
    )
)