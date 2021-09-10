(define (problem cleaning_bathtub_0)
    (:domain igibson)

    (:objects
        sink.n.01_1 - sink.n.01
     	bathtub.n.01_1 - bathtub.n.01
    	soap.n.01_1 - soap.n.01
    	floor.n.01_1 - floor.n.01
    	bucket.n.01_1 - bucket.n.01
    	scrub_brush.n.01_1 - scrub_brush.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (stained bathtub.n.01_1) 
        (onfloor soap.n.01_1 floor.n.01_1) 
        (onfloor bucket.n.01_1 floor.n.01_1) 
        (inside scrub_brush.n.01_1 bathtub.n.01_1) 
        (inroom floor.n.01_1 bathroom) 
        (inroom bathtub.n.01_1 bathroom) 
        (inroom sink.n.01_1 bathroom) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (not 
                (stained ?bathtub.n.01_1)
            )
        )
    )
)