(define (problem washing_curtains_0)
    (:domain igibson)

    (:objects
     	dryer1 - dryer
    	washer1 - washer
    	curtain1 curtain2 curtain3 curtain4 - curtain
    )
    
    (:init 
        (and 
            (ontop dryer1 washer1) 
            (open dryer1) 
            (under washer1 dryer1) 
            (not 
                (open washer1)
            )
        ) 
        (and 
            (inside curtain1 washer1) 
            (not 
                (scrubbed curtain1)
            ) 
            (dusty curtain1)
        ) 
        (and 
            (inside curtain2 washer1) 
            (not 
                (scrubbed curtain2)
            ) 
            (dusty curtain2)
        ) 
        (and 
            (inside curtain3 washer1) 
            (not 
                (scrubbed curtain3)
            ) 
            (dusty curtain3)
        ) 
        (and 
            (inside curtain4 washer1) 
            (not 
                (scrubbed curtain4)
            ) 
            (dusty curtain4)
        ) 
        (inroom dryer1 garage) 
        (inroom washer1 garage)
    )
    
    (:goal 
        (and 
            (and 
                (not 
                    (open ?dryer1)
                ) 
                (open ?washer1)
            ) 
            (forall 
                (?curtain - curtain) 
                (and 
                    (inside ?curtain ?dryer1) 
                    (scrubbed ?curtain) 
                    (not 
                        (dusty ?curtain)
                    )
                )
            )
        )
    )
)