(define (problem sweeping_floors_1)
    (:domain igibson)

    (:objects
     	floor1 floor2 - floor
    	broom1 - broom
    	cabinet1 - cabinet
    	dustpan1 - dustpan
    )
    
    (:init 
        (dusty floor1) 
        (dusty floor2) 
        (and 
            (inside broom1 cabinet1) 
            (inside dustpan1 cabinet1)
        ) 
        (inroom floor1 kitchen) 
        (inroom floor2 livingroom) 
        (inroom cabinet1 kitchen)
    )
    
    (:goal 
        (and 
            (not 
                (dusty ?floor1)
            ) 
            (not 
                (dusty ?floor2)
            ) 
            (not 
                (inside ?broom ?cabinet)
            ) 
            (not 
                (inside ?dustpan ?cabinet)
            )
        )
    )
)