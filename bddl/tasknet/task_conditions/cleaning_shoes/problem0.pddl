(define (problem cleaning_shoes_0)
    (:domain igibson)

    (:objects
     	gym_shoe1 gym_shoe2 - gym_shoe
    	floor1 - floor
    	shoebox1 shoebox2 - shoebox
    	boot1 boot2 - boot
    	rag1 - rag
    	brush1 - brush
    	carpet1 - carpet
    )
    
    (:init 
        (and 
            (dusty gym_shoe1) 
            (ontop gym_shoe1 floor1) 
            (dusty gym_shoe2) 
            (nextto gym_shoe1 gym_shoe2) 
            (ontop gym_shoe2 floor1) 
            (nextto shoebox1 gym_shoe2) 
            (ontop shoebox1 floor1)
        ) 
        (and 
            (dusty boot1) 
            (ontop boot1 floor1) 
            (dusty boot2) 
            (nextto boot2 boot1) 
            (ontop boot2 floor1) 
            (nextto shoebox2 boot2) 
            (ontop shoebox2 floor1)
        ) 
        (and 
            (nextto rag1 shoebox2) 
            (ontop rag1 floor1)
        ) 
        (and 
            (nextto brush1 rag1) 
            (ontop brush1 floor1)
        ) 
        (inroom carpet1 living room) 
        (inroom floor1 living room)
    )
    
    (:goal 
        (and 
            (exists 
                (?shoebox - shoebox) 
                (and 
                    (forall 
                        (?gym_shoe - gym_shoe) 
                        (and 
                            (scrubbed ?gym_shoe) 
                            (inside ?gym_shoe ?shoebox)
                        )
                    ) 
                    (ontop ?shoebox ?floor1)
                )
            ) 
            (exists 
                (?shoebox - shoebox) 
                (and 
                    (forall 
                        (?boot - boot) 
                        (and 
                            (scrubbed ?boot) 
                            (inside ?boot ?shoebox)
                        )
                    ) 
                    (ontop ?shoebox ?floor1)
                )
            ) 
            (exists 
                (?shoebox - shoebox) 
                (and 
                    (not 
                        (scrubbed ?brush1)
                    ) 
                    (nextto ?brush1 ?shoebox) 
                    (not 
                        (scrubbed ?rag1)
                    ) 
                    (nextto ?rag1 ?shoebox)
                )
            )
        )
    )
)