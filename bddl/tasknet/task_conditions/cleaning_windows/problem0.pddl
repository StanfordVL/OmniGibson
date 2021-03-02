(define (problem cleaning_windows_0)
    (:domain igibson)

    (:objects
     	window1 window2 - window
    	rag1 - rag
    	counter1 - counter
    	cleansing_agent1 - cleansing_agent
    	squeegee1 - squeegee
    )
    
    (:init 
        (dusty window1) 
        (not 
            (scrubbed window1)
        ) 
        (open window1) 
        (dusty window2) 
        (not 
            (scrubbed window2)
        ) 
        (open window2) 
        (ontop rag1 counter1) 
        (ontop cleansing_agent1 counter1) 
        (ontop squeegee1 counter1) 
        (inroom window1 kitchen) 
        (inroom window2 kitchen) 
        (inroom counter1 kitchen)
    )
    
    (:goal 
        (and 
            (not 
                (dusty ?window1)
            ) 
            (and 
                (scrubbed ?window1)
            ) 
            (not 
                (open ?window1)
            ) 
            (not 
                (dusty ?window2)
            ) 
            (and 
                (scrubbed ?window2)
            ) 
            (not 
                (open ?window2)
            ) 
            (ontop ?rag1 ?counter1) 
            (soaked ?rag1) 
            (ontop ?squeegee1 ?counter1) 
            (ontop ?cleansing_agent1 ?counter1)
        )
    )
)