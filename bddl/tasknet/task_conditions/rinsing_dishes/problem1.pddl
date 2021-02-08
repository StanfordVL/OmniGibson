(define (problem rinsing_dishes_1)
    (:domain igibson)

    (:objects
     	sink1 sink2 - sink
    	faucet1 - faucet
    	dishrag1 - dishrag
    	soapsuds1 - soapsuds
    	plug1 - plug
    	pan1 pan2 - pan
    	pot1 pot2 - pot
    	dish1 dish2 dish3 dish4 dish5 dish6 dish7 dish8 - dish
    )
    
    (:init 
        (under sink2 faucet1) 
        (inside dishrag1 sink1) 
        (inside soapsuds1 sink1) 
        (inside plug1 sink1) 
        (inside pan1 sink1) 
        (inside pan2 sink1) 
        (inside pot1 sink1) 
        (inside pot2 sink1) 
        (inside dish1 sink1) 
        (inside dish2 sink1) 
        (inside dish3 sink1) 
        (inside dish4 sink1) 
        (inside dish5 sink1) 
        (inside dish6 sink1) 
        (inside dish7 sink1) 
        (inside dish8 sink1) 
        (not 
            (scrubbed dish1)
        ) 
        (not 
            (scrubbed dish2)
        ) 
        (not 
            (scrubbed dish3)
        ) 
        (not 
            (scrubbed dish4)
        ) 
        (not 
            (scrubbed dish5)
        ) 
        (not 
            (scrubbed dish6)
        ) 
        (not 
            (scrubbed dish7)
        ) 
        (not 
            (scrubbed dish8)
        ) 
        (not 
            (scrubbed pan1)
        ) 
        (not 
            (scrubbed pan2)
        ) 
        (not 
            (scrubbed pot1)
        ) 
        (not 
            (scrubbed pot2)
        ) 
        (inroom sink1 kitchen) 
        (inroom sink2 kitchen)
    )
    
    (:goal 
        (and 
            (and 
                (forall 
                    (?dish - dish) 
                    (under ?dish ?faucet1)
                ) 
                (scrubbed ?dish)
            ) 
            (and 
                (forall 
                    (?pan - pan) 
                    (under ?pan ?faucet1)
                ) 
                (scrubbed ?pan)
            ) 
            (and 
                (forall 
                    (?pot - pot) 
                    (under ?pot ?faucet1)
                ) 
                (scrubbed ?pot)
            ) 
            (exists 
                (?sink - sink) 
                (ontop ?dishrag1 ?sink)
            ) 
            (exists 
                (?sink - sink) 
                (ontop ?plug1 ?sink)
            ) 
            (forall 
                (?sink - sink) 
                (not 
                    (inside ?soapsuds1 ?sink)
                )
            )
        )
    )
)