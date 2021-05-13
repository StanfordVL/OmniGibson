(define (problem cleaning_bathtub_1)
    (:domain igibson)

    (:objects
     	soap.n.01_1 - soap.n.01
    	bathtub.n.01_1 - bathtub.n.01
    	rag.n.01_1 - rag.n.01
    	bucket.n.01_1 - bucket.n.01
    	scrub_brush.n.01_1 - scrub_brush.n.01
    	floor.n.01_1 - floor.n.01
        agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (inside soap.n.01_1 bathtub.n.01_1) 
        (inside rag.n.01_1 bathtub.n.01_1) 
        (inside bucket.n.01_1 bathtub.n.01_1) 
        (inside scrub_brush.n.01_1 bathtub.n.01_1) 
        (inroom floor.n.01_1 bathroom) 
        (inroom bathtub.n.01_1 bathroom) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (inside ?soap.n.01_1 ?bucket.n.01_1) 
            (inside ?rag.n.01_1 ?bucket.n.01_1) 
            (onfloor ?bucket.n.01_1 ?floor.n.01_1) 
            (inside ?scrub_brush.n.01_1 ?bucket.n.01_1)
        )
    )
)