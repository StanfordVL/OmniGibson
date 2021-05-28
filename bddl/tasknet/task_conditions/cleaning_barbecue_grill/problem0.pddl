(define (problem cleaning_barbecue_grill_0)
    (:domain igibson)

    (:objects
     	grill.n.02_1 - grill.n.02
    	floor.n.01_1 - floor.n.01
    	rag.n.01_1 - rag.n.01
    	bucket.n.01_1 - bucket.n.01
        table.n.02_1 - table.n.02
        sink.n.01_1 - sink.n.01
        agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor grill.n.02_1 floor.n.01_1) 
        (stained grill.n.02_1) 
        (dusty grill.n.02_1) 
        (ontop bucket.n.01_1 table.n.02_1) 
        (ontop rag.n.01_1 table.n.02_1) 
        (inroom floor.n.01_1 garage) 
        (inroom table.n.02_1 storage_room) 
        (inroom sink.n.01_1 storage_room)
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (not 
                (stained ?grill.n.02_1)
            ) 
            (not 
                (dusty ?grill.n.02_1)
            )
        )
    )
)