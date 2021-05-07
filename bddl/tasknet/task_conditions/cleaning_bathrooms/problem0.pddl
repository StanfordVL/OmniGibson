(define (problem cleaning_bathrooms_0)
    (:domain igibson)

    (:objects
     	rag.n.01_1 - rag.n.01
    	sink.n.01_1 - sink.n.01
    	soap.n.01_1 - soap.n.01
    	bucket.n.01_1 - bucket.n.01
    	bathtub.n.01_1 - bathtub.n.01
    	floor.n.01_1 - floor.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (inside rag.n.01_1 sink.n.01_1) 
        (ontop soap.n.01_1 bucket.n.01_1) 
        (dusty bathtub.n.01_1) 
        (dusty sink.n.01_1) 
        (dusty floor.n.01_1) 
        (onfloor bucket.n.01_1 floor.n.01_1) 
        (inroom floor.n.01_1 bathroom) 
        (inroom bathtub.n.01_1 bathroom) 
        (inroom sink.n.01_1 bathroom) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (inside ?soap.n.01_1 ?bucket.n.01_1) 
            (onfloor ?bucket.n.01_1 ?floor.n.01) 
            (inside ?rag.n.01_1 ?bucket.n.01_1)
        )
    )
)