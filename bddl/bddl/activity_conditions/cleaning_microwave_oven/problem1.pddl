(define (problem cleaning_microwave_oven_1)
    (:domain igibson)

    (:objects
     	microwave1 - microwave
    	counter1 - counter
    	floor1 - floor
    	crumb1 crumb2 crumb3 crumb4 crumb5 - crumb
    	soap1 - soap
    	scrub_brush1 - scrub_brush
    	popcorn1 - popcorn
    	receptacle1 - receptacle
    )
    
    (:init 
        (dusty microwave1) 
        (inside crumb1 microwave1) 
        (inside crumb2 microwave1) 
        (inside crumb3 microwave1) 
        (inside crumb4 microwave1) 
        (inside crumb5 microwave1) 
        (inside soap1 microwave1) 
        (ontop scrub_brush1 counter1) 
        (inside popcorn1 microwave1) 
        (ontop receptacle1 floor1) 
        (inroom microwave1 kitchen) 
        (inroom floor1 kitchen) 
        (inroom counter1 kitchen)
    )
    
    (:goal 
        (and 
            (forall 
                (?crumb - crumb) 
                (inside ?crumb ?receptacle1)
            ) 
            (inside ?popcorn1 ?receptacle1) 
            (scrubbed ?microwave1) 
            (ontop ?scrub_brush1 ?counter1) 
            (ontop ?receptacle1 ?floor1)
        )
    )
)