(define (problem stripping_paint_1)
    (:domain igibson)

    (:objects
     	scraper1 - scraper
    	floor1 - floor
    	coat_of_paint1 coat_of_paint2 coat_of_paint3 - coat_of_paint
    	stool1 - stool
    	table1 - table
    	bench1 - bench
    )
    
    (:init 
        (ontop scraper1 floor1) 
        (ontop coat_of_paint1 stool1) 
        (ontop coat_of_paint3 table1) 
        (ontop coat_of_paint2 bench1) 
        (inroom table1 garage) 
        (inroom bench1 garage) 
        (inroom stool1 garage) 
        (inroom floor1 garage)
    )
    
    (:goal 
        (and 
            (forall 
                (?coat_of_paint - coat_of_paint) 
                (and 
                    (not 
                        (ontop ?coat_of_paint ?bench1)
                    ) 
                    (not 
                        (ontop ?coat_of_paint ?stool1)
                    ) 
                    (not 
                        (ontop ?coat_of_paint ?table1)
                    )
                )
            ) 
            (ontop ?scraper1 ?bench1)
        )
    )
)