(define (problem cleaning_floors_1)
    (:domain igibson)

    (:objects
     	floor1 floor2 - floor
    	broom1 - broom
    	oven1 - oven
    	dustpan1 - dustpan
    	fridge1 - fridge
    	squeegee1 - squeegee
    	bucket1 - bucket
    	sink1 - sink
    	soapsuds1 - soapsuds
    	table1 - table
    	sofa1 - sofa
    )
    
    (:init 
        (not 
            (scrubbed floor1)
        ) 
        (not 
            (scrubbed floor2)
        ) 
        (nextto broom1 oven1) 
        (ontop dustpan1 fridge1) 
        (nextto squeegee1 fridge1) 
        (nextto bucket1 sink1) 
        (and 
            (nextto soapsuds1 sink1) 
            (inside soapsuds1 bucket1)
        ) 
        (inroom table1 kitchen) 
        (inroom sink1 kitchen) 
        (inroom sofa1 livingroom) 
        (inroom oven1 kitchen) 
        (inroom floor1 kitchen) 
        (inroom floor2 livingroom) 
        (inroom fridge1 kitchen)
    )
    
    (:goal 
        (and 
            (forall 
                (?floor - floor) 
                (scrubbed ?floor)
            ) 
            (nextto ?broom1 ?oven1) 
            (nextto ?dustpan1 ?oven1) 
            (and 
                (nextto ?bucket1 ?fridge1) 
                (not 
                    (inside ?soapsuds1 ?bucket1)
                )
            ) 
            (and 
                (nextto ?squeegee1 ?fridge1) 
                (inside ?squeegee1 ?bucket1)
            )
        )
    )
)