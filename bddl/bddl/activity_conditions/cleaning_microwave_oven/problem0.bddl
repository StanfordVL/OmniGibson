(define (problem cleaning_microwave_oven_0)
    (:domain igibson)

    (:objects
     	microwave.n.02_1 - microwave.n.02
    	rag.n.01_1 - rag.n.01
    	countertop.n.01_1 - countertop.n.01
    	ashcan.n.01_1 - ashcan.n.01
    	floor.n.01_1 - floor.n.01
    	cabinet.n.01_1 - cabinet.n.01
    	sink.n.01_1 - sink.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (dusty microwave.n.02_1) 
        (stained microwave.n.02_1) 
        (ontop rag.n.01_1 countertop.n.01_1) 
        (onfloor ashcan.n.01_1 floor.n.01_1) 
        (inroom microwave.n.02_1 kitchen) 
        (inroom cabinet.n.01_1 kitchen) 
        (inroom countertop.n.01_1 kitchen) 
        (inroom sink.n.01_1 kitchen) 
        (inroom floor.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (not 
                (dusty ?microwave.n.02_1)
            ) 
            (not 
                (stained ?microwave.n.02_1)
            )
        )
    )
)