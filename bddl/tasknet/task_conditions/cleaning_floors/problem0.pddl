(define (problem cleaning_floors_0)
    (:domain igibson)

    (:objects
     	soap1 soap2 - soap
    	cabinet1 cabinet2 - cabinet
    	bucket1 bucket2 - bucket
    	broom1 - broom
    	dustpan1 - dustpan
    	rag1 rag2 - rag
    	floor1 floor2 - floor
    	sink1 sink2 - sink
    )
    
    (:init 
        (inside soap1 cabinet1) 
        (inside soap2 cabinet2) 
        (inside bucket1 cabinet1) 
        (inside bucket2 cabinet2) 
        (nextto broom1 cabinet2) 
        (nextto dustpan1 cabinet2) 
        (inside rag1 cabinet1) 
        (inside rag2 cabinet2) 
        (not 
            (scrubbed floor1)
        ) 
        (not 
            (scrubbed floor2)
        ) 
        (dusty floor2) 
        (inroom floor1 bathroom) 
        (inroom floor2 kitchen) 
        (inroom sink1 bathroom) 
        (inroom sink2 kitchen) 
        (inroom cabinet1 bathroom) 
        (inroom cabinet2 kitchen)
    )
    
    (:goal 
        (and 
            (scrubbed ?floor1) 
            (not 
                (dusty ?floor2)
            ) 
            (scrubbed ?floor2) 
            (nextto ?broom1 ?cabinet2) 
            (nextto ?dustpan1 ?cabinet2) 
            (forall 
                (?rag - rag) 
                (exists 
                    (?sink - sink) 
                    (inside ?rag ?sink)
                )
            ) 
            (forpairs 
                (?bucket - bucket) 
                (?cabinet - cabinet) 
                (inside ?bucket ?cabinet)
            ) 
            (soaked ?rag1) 
            (soaked ?rag2) 
            (forpairs 
                (?soap - soap) 
                (?cabinet - cabinet) 
                (inside ?soap ?cabinet)
            )
        )
    )
)