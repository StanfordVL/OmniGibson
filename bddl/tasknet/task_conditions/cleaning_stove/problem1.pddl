(define (problem cleaning_stove_1)
    (:domain igibson)

    (:objects
     	rag1 - rag
    	cabinet1 - cabinet
    	soap1 - soap
    	sink1 - sink
    	sauce1 - sauce
    	stove1 - stove
    	pan1 - pan
    	pot1 pot2 - pot
    	towel_rack1 - towel_rack
    	range_hood1 - range_hood
    )
    
    (:init 
        (not 
            (scrubbed stove1)
        ) 
        (dusty range_hood1) 
        (not 
            (soaked rag1)
        ) 
        (inside rag1 cabinet1) 
        (nextto soap1 sink1) 
        (ontop sauce1 stove1) 
        (and 
            (not 
                (scrubbed pan1)
            ) 
            (not 
                (scrubbed pot1)
            ) 
            (not 
                (scrubbed pot2)
            )
        ) 
        (ontop pot1 stove1) 
        (ontop pan1 stove1) 
        (ontop pot2 stove1) 
        (inroom stove1 kitchen) 
        (inroom towel_rack1 kitchen) 
        (inroom cabinet1 kitchen) 
        (inroom sink1 kitchen) 
        (inroom range_hood1 kitchen)
    )
    
    (:goal 
        (and 
            (ontop ?rag1 ?towel_rack1) 
            (soaked ?rag1) 
            (not 
                (nextto ?soap1 ?sink1)
            ) 
            (scrubbed ?stove1) 
            (not 
                (dusty ?range_hood1)
            ) 
            (not 
                (ontop ?sauce1 ?stove1)
            ) 
            (and 
                (inside ?pot1 ?sink1) 
                (inside ?pot2 ?sink1) 
                (inside ?pan1 ?sink1)
            )
        )
    )
)