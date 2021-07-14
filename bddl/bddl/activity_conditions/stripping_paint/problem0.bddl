(define (problem stripping_paint_0)
    (:domain igibson)

    (:objects
     	floor1 - floor
    	coat_of_paint1 coat_of_paint2 coat_of_paint3 - coat_of_paint
    	table1 - table
    	chair1 - chair
    	console_table1 - console_table
    	rag1 - rag
    	scraper1 - scraper
    )
    
    (:init 
        (and 
            (and 
                (ontop table1 floor1) 
                (ontop coat_of_paint1 table1)
            ) 
            (and 
                (ontop chair1 floor1) 
                (ontop coat_of_paint2 chair1)
            ) 
            (and 
                (ontop console_table1 floor1) 
                (ontop coat_of_paint3 console_table1)
            )
        ) 
        (ontop rag1 floor1) 
        (ontop scraper1 floor1) 
        (inroom table1 garage) 
        (inroom chair1 garage) 
        (inroom floor1 garage) 
        (inroom console_table1 garage)
    )
    
    (:goal 
        (and 
            (forall 
                (?coat_of_paint - coat_of_paint) 
                (and 
                    (or 
                        (ontop ?coat_of_paint ?floor1) 
                        (ontop ?coat_of_paint ?rag1)
                    ) 
                    (not 
                        (ontop ?coat_of_paint ?table1)
                    ) 
                    (not 
                        (ontop ?coat_of_paint ?chair1)
                    ) 
                    (not 
                        (ontop ?coat_of_paint ?console_table1)
                    )
                )
            ) 
            (and 
                (ontop ?table1 ?floor1) 
                (scrubbed ?table1)
            ) 
            (and 
                (ontop ?chair1 ?floor1) 
                (scrubbed ?chair1)
            ) 
            (ontop ?console_table1 ?floor1) 
            (ontop ?scraper1 ?floor1)
        )
    )
)