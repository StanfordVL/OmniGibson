(define (problem mopping_floors_0)
    (:domain igibson)

    (:objects
     	broom.n.01_1 - broom.n.01
    	floor.n.01_1 floor.n.01_2 - floor.n.01
    	soap.n.01_1 - soap.n.01
    	bucket.n.01_1 - bucket.n.01
    	piece_of_cloth.n.01_1 - piece_of_cloth.n.01
    	sink.n.01_1 - sink.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor broom.n.01_1 floor.n.01_2) 
        (onfloor soap.n.01_1 floor.n.01_1) 
        (onfloor bucket.n.01_1 floor.n.01_1) 
        (onfloor piece_of_cloth.n.01_1 floor.n.01_1) 
        (stained floor.n.01_1) 
        (inroom floor.n.01_1 bathroom) 
        (inroom floor.n.01_2 corridor) 
        (inroom sink.n.01_1 bathroom) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (not 
                (stained ?floor.n.01_1)
            ) 
            (nextto ?bucket.n.01_1 ?sink.n.01_1) 
            (inside ?soap.n.01_1 ?bucket.n.01_1) 
            (nextto ?broom.n.01_1 ?sink.n.01_1)
        )
    )
)