(define (problem vacuuming_floors_1)
    (:domain igibson)

    (:objects
     	carpet1 carpet2 - carpet
    	bed1 - bed
    	chest1 - chest
    	sofa1 - sofa
    	coffee_table1 - coffee_table
    	tv1 - tv
    	cabinet1 - cabinet
    )
    
    (:init 
        (under carpet1 bed1) 
        (and 
            (under carpet1 chest1) 
            (dusty carpet1)
        ) 
        (under carpet2 sofa1) 
        (and 
            (under carpet2 coffee_table1) 
            (under carpet2 tv1) 
            (dusty carpet2)
        ) 
        (inside ?vacuum cabinet1) 
        (inroom carpet1 bedroom) 
        (inroom carpet2 livingroom) 
        (inroom bed1 bedroom) 
        (inroom chest1 bedroom) 
        (inroom sofa1 livingroom) 
        (inroom coffee_table1 livingroom) 
        (inroom tv1 livingroom) 
        (inroom cabinet1 bedroom)
    )
    
    (:goal 
        (and 
            (not 
                (dusty ?carpet1)
            ) 
            (not 
                (dusty ?carpet2)
            ) 
            (inside ?vacuum1 ?cabinet1)
        )
    )
)