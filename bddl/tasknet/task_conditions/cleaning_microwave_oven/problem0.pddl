(define (problem cleaning_microwave_oven_0)
    (:domain igibson)

    (:objects
     	microwave1 - microwave
    	counter1 - counter
    	sink1 - sink
    	garbage1 - garbage
    	crumb1 crumb2 crumb3 - crumb
    	lemon1 - lemon
    	vinegar1 - vinegar
    	disinfectant1 - disinfectant
    	water1 - water
    	soap1 - soap
    	washcloth1 - washcloth
    )
    
    (:init 
        (and 
            (ontop microwave1 counter1) 
            (dusty microwave1)
        ) 
        (nextto sink1 counter1) 
        (nextto garbage1 sink1) 
        (and 
            (inside crumb1 microwave1) 
            (inside crumb2 microwave1) 
            (inside crumb3 microwave1)
        ) 
        (and 
            (ontop lemon1 microwave1) 
            (ontop vinegar1 microwave1)
        ) 
        (and 
            (ontop disinfectant1 sink1) 
            (inside water1 sink1) 
            (inside soap1 sink1) 
            (and 
                (ontop washcloth1 sink1) 
                (not 
                    (soaked washcloth1)
                )
            )
        ) 
        (inroom counter1 kitchen) 
        (inroom sink1 kitchen) 
        (inroom microwave1 kitchen)
    )
    
    (:goal 
        (and 
            (scrubbed ?microwave1) 
            (and 
                (soaked ?washcloth1) 
                (dusty ?washcloth1)
            ) 
            (forall 
                (?crumb - crumb) 
                (inside ?crumb ?garbage1)
            ) 
            (inside ?lemon1 ?garbage1)
        )
    )
)