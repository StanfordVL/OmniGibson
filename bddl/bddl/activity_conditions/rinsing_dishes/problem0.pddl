(define (problem rinsing_dishes_0)
    (:domain igibson)

    (:objects
     	soapsuds1 soapsuds10 soapsuds11 soapsuds12 soapsuds2 soapsuds3 soapsuds4 soapsuds5 soapsuds6 soapsuds7 soapsuds8 soapsuds9 - soapsuds
    	pot1 - pot
    	sink1 - sink
    	cup1 cup2 cup3 - cup
    	bowl1 bowl2 bowl3 - bowl
    	spoon1 spoon2 spoon3 - spoon
    	dish1 dish2 - dish
    	dishrag1 - dishrag
    	cabinet1 cabinet2 - cabinet
    	water1 - water
    )
    
    (:init 
        (and 
            (and 
                (ontop soapsuds1 pot1) 
                (inside pot1 sink1) 
                (under sink1 soapsuds1)
            ) 
            (and 
                (inside soapsuds2 cup1) 
                (inside cup1 sink1) 
                (under sink1 soapsuds2)
            ) 
            (and 
                (inside soapsuds3 cup2) 
                (inside cup2 sink1) 
                (under sink1 soapsuds3)
            ) 
            (and 
                (inside soapsuds4 cup3) 
                (inside cup3 sink1) 
                (under sink1 soapsuds4)
            ) 
            (and 
                (inside soapsuds5 bowl1) 
                (inside bowl1 sink1) 
                (under sink1 soapsuds5)
            ) 
            (and 
                (inside soapsuds6 bowl2) 
                (inside bowl2 sink1) 
                (under sink1 soapsuds6)
            ) 
            (and 
                (inside soapsuds7 bowl3) 
                (inside bowl3 sink1) 
                (under sink1 soapsuds7)
            ) 
            (and 
                (ontop soapsuds8 spoon1) 
                (inside spoon1 sink1) 
                (under sink1 soapsuds8)
            ) 
            (and 
                (ontop soapsuds9 spoon2) 
                (inside spoon2 sink1) 
                (under sink1 soapsuds9)
            ) 
            (and 
                (ontop soapsuds10 spoon3) 
                (inside spoon3 sink1) 
                (under sink1 soapsuds10)
            ) 
            (and 
                (ontop soapsuds11 dish1) 
                (inside dish1 sink1) 
                (under sink1 soapsuds11)
            ) 
            (and 
                (ontop soapsuds12 dish2) 
                (inside dish2 sink1) 
                (under sink1 soapsuds12)
            )
        ) 
        (and 
            (nextto dishrag1 sink1) 
            (not 
                (soaked dishrag1)
            )
        ) 
        (nextto cabinet1 sink1) 
        (inroom sink1 kitchen) 
        (inroom cabinet1 kitchen) 
        (inroom cabinet2 kitchen)
    )
    
    (:goal 
        (and 
            (forall 
                (?soapsuds - soapsuds) 
                (inside ?soapsuds ?sink1)
            ) 
            (exists 
                (?cabinet - cabinet) 
                (forall 
                    (?cup - cup) 
                    (and 
                        (inside ?cup ?cabinet) 
                        (scrubbed ?cup)
                    )
                )
            ) 
            (exists 
                (?cabinet - cabinet) 
                (forall 
                    (?bowl - bowl) 
                    (and 
                        (inside ?bowl?cabinet) 
                        (scrubbed ?bowl)
                    )
                )
            ) 
            (exists 
                (?cabinet - cabinet) 
                (forall 
                    (?spoon - spoon) 
                    (and 
                        (inside ?spoon ?cabinet) 
                        (scrubbed ?spoon)
                    )
                )
            ) 
            (exists 
                (?cabinet - cabinet) 
                (forall 
                    (?dish - dish) 
                    (and 
                        (inside ?dish ?dish) 
                        (scrubbed ?dish)
                    )
                )
            ) 
            (exists 
                (?cabinet - cabinet) 
                (and 
                    (inside ?pot1 ?cabinet) 
                    (scrubbed ?pot1)
                )
            )
        )
    )
)