(define (problem cleaning_oven_0)
    (:domain igibson)

    (:objects
     	receptacle.n.01_1 - receptacle.n.01
    	floor.n.01_1 - floor.n.01
    	soap.n.01_1 - soap.n.01
    	cabinet.n.01_1 - cabinet.n.01
    	rag.n.01_1 rag.n.01_2 - rag.n.01
    	newspaper.n.03_1 - newspaper.n.03
    	sink.n.01_1 - sink.n.01
    	scrub_brush.n.01_1 - scrub_brush.n.01
    	oven.n.01_1 - oven.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor receptacle.n.01_1 floor.n.01_1) 
        (inside soap.n.01_1 cabinet.n.01_1) 
        (inside rag.n.01_1 cabinet.n.01_1) 
        (inside rag.n.01_2 cabinet.n.01_1) 
        (not 
            (soaked rag.n.01_1)
        ) 
        (not 
            (soaked rag.n.01_2)
        ) 
        (onfloor newspaper.n.03_1 floor.n.01_1) 
        (inside scrub_brush.n.01_1 cabinet.n.01_1) 
        (not 
            (soaked scrub_brush.n.01_1)
        ) 
        (stained oven.n.01_1) 
        (inroom oven.n.01_1 kitchen) 
        (inroom cabinet.n.01_1 kitchen) 
        (inroom sink.n.01_1 kitchen) 
        (inroom floor.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forall 
                (?rag.n.01 - rag.n.01) 
                (soaked ?rag.n.01)
            ) 
            (soaked ?scrub_brush.n.01_1) 
            (not 
                (stained ?oven.n.01_1)
            )
        )
    )
)