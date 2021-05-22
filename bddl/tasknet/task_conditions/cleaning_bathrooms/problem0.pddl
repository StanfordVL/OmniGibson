(define (problem cleaning_bathrooms_0)
    (:domain igibson)

    (:objects
        sink.n.01_1 - sink.n.01
        bathtub.n.01_1 - bathtub.n.01
        toilet.n.02_1 - toilet.n.02
        floor.n.01_1 - floor.n.01
        bucket.n.01_1 - bucket.n.01
        soap.n.01_1 - soap.n.01
        brush.n.02_1 - brush.n.02
        rag.n.01_1 - rag.n.01
        agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (stained sink.n.01_1) 
        (stained bathtub.n.01_1) 
        (stained toilet.n.02_1) 
        (stained floor.n.01_1) 
        (inside soap.n.01_1 sink.n.01_1) 
        (inside brush.n.02_1 bathtub.n.01_1) 
        (onfloor bucket.n.01_1 floor.n.01_1) 
        (not 
            (soaked rag.n.01_1)
        ) 
        (onfloor rag.n.01_1 floor.n.01_1) 
        (onfloor agent.n.01_1 floor.n.01_1) 
        (inroom floor.n.01_1 bathroom) 
        (inroom toilet.n.02_1 bathroom) 
        (inroom sink.n.01_1 bathroom) 
        (inroom bathtub.n.01_1 bathroom)
    )
    
    (:goal 
        (and 
            (not 
                (stained ?toilet.n.02_1)
            ) 
            (not 
                (stained ?bathtub.n.01_1)
            ) 
            (not 
                (stained ?sink.n.01_1)
            ) 
            (not 
                (stained ?floor.n.01_1)
            ) 
            (and 
                (soaked ?rag.n.01_1) 
                (inside ?rag.n.01_1 ?bucket.n.01_1)
            )
        )
    )
)