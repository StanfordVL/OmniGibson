(define (problem cleaning_bathrooms_0)
    (:domain igibson)

    (:objects
     	bag1 - bag
    	shelf1 shelf2 - shelf
    	rag1 rag2 - rag
    	toothpaste1 - toothpaste
    	sink1 - sink
    	toothbrush1 toothbrush2 - toothbrush
    	counter1 - counter
    	towel1 towel2 - towel
    	shower1 - shower
    	soap1 - soap
    	garbage1 - garbage
    	floor1 - floor
    	bucket1 - bucket
    	razor1 - razor
    	mirror1 - mirror
    	toilet1 - toilet
    	towel_rack1 towel_rack2 - towel_rack
    )
    
    (:init 
        (ontop bag1 shelf2) 
        (ontop rag1 shelf2) 
        (ontop rag2 shelf2) 
        (nextto toothpaste1 sink1) 
        (inside toothbrush1 sink1) 
        (ontop toothbrush2 counter1) 
        (nextto towel1 sink1) 
        (nextto towel2 shower1) 
        (ontop soap1 shelf2) 
        (ontop garbage1 floor1) 
        (under bucket1 sink1) 
        (ontop razor1 counter1) 
        (not 
            (soaked rag1)
        ) 
        (not 
            (soaked rag2)
        ) 
        (not 
            (scrubbed floor1)
        ) 
        (dusty mirror1) 
        (not 
            (scrubbed shower1)
        ) 
        (not 
            (scrubbed toilet1)
        ) 
        (not 
            (scrubbed sink1)
        ) 
        (not 
            (scrubbed counter1)
        ) 
        (inroom floor1 bathroom) 
        (inroom mirror1 bathroom) 
        (inroom shower1 bathroom) 
        (inroom toilet1 bathroom) 
        (inroom sink1 bathroom) 
        (inroom shelf1 bathroom) 
        (inroom shelf2 bathroom) 
        (inroom towel_rack1 bathroom) 
        (inroom towel_rack2 bathroom) 
        (inroom counter1 bathroom)
    )
    
    (:goal 
        (and 
            (scrubbed ?floor1) 
            (not 
                (dusty ?mirror1)
            ) 
            (scrubbed ?shower1) 
            (scrubbed ?toilet1) 
            (scrubbed ?sink1) 
            (forall 
                (?rag - rag) 
                (soaked ?rag)
            ) 
            (forpairs 
                (?towel - towel) 
                (?towel_rack - towel_rack) 
                (ontop ?towel ?towel_rack)
            ) 
            (exists 
                (?shelf - shelf) 
                (and 
                    (forall 
                        (?toothbrush - toothbrush) 
                        (ontop ?toothbrush ?shelf)
                    ) 
                    (ontop ?toothpaste1 ?shelf) 
                    (ontop ?razor1 ?shelf)
                )
            ) 
            (forall 
                (?rag - rag) 
                (inside ?rag ?bucket1)
            ) 
            (inside ?bag1 ?garbage1) 
            (exists 
                (?shelf - shelf) 
                (and 
                    (ontop ?soap1 ?shelf) 
                    (ontop ?bucket1 ?shelf)
                )
            ) 
            (imply 
                (ontop ?razor1 ?shelf1) 
                (ontop ?bucket1 ?shelf2)
            ) 
            (scrubbed ?counter1)
        )
    )
)