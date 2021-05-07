(define (problem cleaning_oven_0)
    (:domain igibson)

    (:objects
     	oven.n.01_1 - oven.n.01
    	soap.n.01_1 - soap.n.01
    	floor.n.01_1 - floor.n.01
    	rag.n.01_1 - rag.n.01
    	sink.n.01_1 - sink.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (stained oven.n.01_1) 
        (dusty oven.n.01_1) 
        (ontop soap.n.01_1 oven.n.01_1) 
        (onfloor rag.n.01_1 floor.n.01_1) 
        (not 
            (stained rag.n.01_1)
        ) 
        (inroom sink.n.01_1 kitchen) 
        (inroom oven.n.01_1 kitchen) 
        (inroom floor.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (not 
                (stained ?oven.n.01_1)
            ) 
            (not 
                (dusty ?oven.n.01_1)
            ) 
        )
    )
)