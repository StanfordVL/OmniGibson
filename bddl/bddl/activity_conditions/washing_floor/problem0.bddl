(define (problem washing_floor_0)
    (:domain igibson)

    (:objects
     	bucket.n.01_1 - bucket.n.01
    	floor.n.01_1 - floor.n.01
    	soap.n.01_1 - soap.n.01
    	towel.n.01_1 - towel.n.01
    	shower.n.01_1 - shower.n.01
    	toilet.n.02_1 - toilet.n.02
    	bed.n.01_1 - bed.n.01
    	sink.n.01_1 - sink.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor bucket.n.01_1 floor.n.01_1) 
        (ontop soap.n.01_1 towel.n.01_1) 
        (onfloor soap.n.01_1 floor.n.01_1) 
        (onfloor towel.n.01_1 floor.n.01_1) 
        (not 
            (stained towel.n.01_1)
        ) 
        (dusty floor.n.01_1) 
        (stained floor.n.01_1) 
        (inroom floor.n.01_1 bathroom) 
        (inroom shower.n.01_1 bathroom) 
        (inroom toilet.n.02_1 bathroom) 
        (inroom bed.n.01_1 bedroom) 
        (inroom sink.n.01_1 bathroom) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (not 
                (or 
                    (dusty ?floor.n.01_1) 
                    (stained ?floor.n.01_1)
                )
            )
        )
    )
)