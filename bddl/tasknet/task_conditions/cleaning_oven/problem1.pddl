(define (problem cleaning_oven_1)
    (:domain igibson)

    (:objects
     	garbage1 - garbage
    	floor1 - floor
    	oven1 - oven
    	wall1 - wall
    	bowl1 bowl2 bowl3 - bowl
    	rag1 - rag
    	sink1 - sink
    	pan1 pan2 - pan
    	crumb1 crumb2 crumb3 crumb4 crumb5 crumb6 crumb7 - crumb
    	cabinet1 cabinet2 - cabinet
    )
    
    (:init 
        (ontop garbage1 floor1) 
        (and 
            (inside oven1 wall1) 
            (dusty oven1)
        ) 
        (and 
            (inside bowl1 oven1) 
            (dusty bowl1)
        ) 
        (and 
            (inside bowl2 oven1) 
            (dusty bowl2)
        ) 
        (and 
            (inside bowl3 oven1) 
            (dusty bowl3)
        ) 
        (and 
            (inside rag1 sink1) 
            (soaked rag1)
        ) 
        (and 
            (inside pan1 oven1) 
            (dusty pan1)
        ) 
        (and 
            (inside pan2 oven1) 
            (dusty pan2)
        ) 
        (inside crumb1 oven1) 
        (inside crumb2 oven1) 
        (inside crumb3 oven1) 
        (inside crumb4 oven1) 
        (inside crumb5 oven1) 
        (inside crumb6 oven1) 
        (inside crumb7 oven1) 
        (inroom oven1 kitchen) 
        (inroom cabinet1 kitchen) 
        (inroom cabinet2 kitchen) 
        (inroom sink1 kitchen) 
        (inroom floor1 kitchen)
    )
    
    (:goal 
        (and 
            (exists 
                (?cabinet - cabinet) 
                (and 
                    (forall 
                        (?pan - pan) 
                        (and 
                            (inside ?pan ?cabinet) 
                            (scrubbed ?pan)
                        )
                    ) 
                    (forall 
                        (?bowl - bowl) 
                        (not 
                            (inside ?bowl ?cabinet)
                        )
                    )
                )
            ) 
            (exists 
                (?cabinet - cabinet) 
                (and 
                    (forall 
                        (?bowl - bowl) 
                        (and 
                            (inside ?bowl ?cabinet) 
                            (scrubbed ?bowl)
                        )
                    ) 
                    (forall 
                        (?pan - pan) 
                        (not 
                            (inside ?pan ?cabinet)
                        )
                    )
                )
            ) 
            (forall 
                (?crumb - crumb) 
                (inside ?crumb ?garbage1)
            ) 
            (inside ?rag1 ?sink1) 
            (scrubbed ?oven1)
        )
    )
)