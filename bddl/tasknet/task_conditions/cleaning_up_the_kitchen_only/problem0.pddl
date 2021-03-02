(define (problem cleaning_up_the_kitchen_only_0)
    (:domain igibson)

    (:objects
     	dishwasher1 - dishwasher
    	counter1 - counter
    	floor1 - floor
    	plate1 plate2 plate3 plate4 plate5 - plate
    	sink1 - sink
    	cooktop1 - cooktop
    	mold1 mold2 mold3 mold4 - mold
    	bucket1 - bucket
    	receptacle1 - receptacle
    	soapsuds1 - soapsuds
    )
    
    (:init 
        (nextto dishwasher1 counter1) 
        (ontop dishwasher1 floor1) 
        (inside plate1 sink1) 
        (inside plate2 sink1) 
        (inside plate3 sink1) 
        (inside plate4 sink1) 
        (inside plate5 sink1) 
        (nextto sink1 cooktop1) 
        (ontop mold2 cooktop1) 
        (ontop mold3 cooktop1) 
        (ontop mold4 cooktop1) 
        (ontop bucket1 floor1) 
        (ontop receptacle1 floor1) 
        (dusty counter1) 
        (dusty cooktop1) 
        (dusty floor1) 
        (ontop soapsuds1 counter1) 
        (ontop mold1 cooktop1) 
        (ontop cooktop1 floor1) 
        (ontop counter1 floor1) 
        (inroom cooktop1 kitchen) 
        (inroom floor1 kitchen) 
        (inroom counter1 kitchen) 
        (inroom sink1 kitchen) 
        (inroom dishwasher1 kitchen)
    )
    
    (:goal 
        (and 
            (scrubbed ?sink1) 
            (forn 
                (5) 
                (?plate - plate) 
                (inside ?plate ?dishwasher1)
            ) 
            (inside ?mold2 ?receptacle1) 
            (inside ?mold3 ?receptacle1) 
            (inside ?mold4 ?receptacle1) 
            (scrubbed ?cooktop1) 
            (scrubbed ?floor1) 
            (scrubbed ?counter1) 
            (ontop ?cooktop1 ?floor1) 
            (ontop ?counter1 ?floor1) 
            (ontop ?receptacle1 ?floor1) 
            (ontop ?bucket1 ?floor1) 
            (inside ?soapsuds1 ?bucket1) 
            (inside ?mold1 ?receptacle1)
        )
    )
)