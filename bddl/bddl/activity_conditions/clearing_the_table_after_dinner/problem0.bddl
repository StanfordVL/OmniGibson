(define (problem clearing_the_table_after_dinner_0)
    (:domain igibson)

    (:objects
     	floor.n.01_1 - floor.n.01
    	chair.n.01_1 chair.n.01_2 - chair.n.01
    	table.n.02_1 - table.n.02
    	cup.n.01_1 cup.n.01_2 - cup.n.01
    	bucket.n.01_1 bucket.n.01_2 - bucket.n.01
    	bowl.n.01_1 bowl.n.01_2 bowl.n.01_3 bowl.n.01_4 - bowl.n.01
    	catsup.n.01_1 - catsup.n.01
    	beverage.n.01_1 beverage.n.01_2 - beverage.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (ontop cup.n.01_1 table.n.02_1) 
        (ontop cup.n.01_2 table.n.02_1) 
        (onfloor bucket.n.01_1 floor.n.01_1) 
        (ontop bowl.n.01_1 table.n.02_1) 
        (ontop bowl.n.01_2 table.n.02_1) 
        (ontop bowl.n.01_3 table.n.02_1) 
        (ontop bowl.n.01_4 table.n.02_1) 
        (ontop catsup.n.01_1 table.n.02_1) 
        (ontop beverage.n.01_1 table.n.02_1) 
        (onfloor beverage.n.01_2 floor.n.01_1) 
        (onfloor bucket.n.01_2 floor.n.01_1) 
        (inroom floor.n.01_1 dining_room) 
        (inroom chair.n.01_1 dining_room) 
        (inroom chair.n.01_2 dining_room) 
        (inroom table.n.02_1 dining_room) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forall 
                (?cup.n.01 - cup.n.01) 
                (exists 
                    (?bucket.n.01 - bucket.n.01) 
                    (inside ?cup.n.01 ?bucket.n.01)
                )
            ) 
            (forall 
                (?bowl.n.01 - bowl.n.01) 
                (exists 
                    (?bucket.n.01 - bucket.n.01) 
                    (inside ?bowl.n.01 ?bucket.n.01)
                )
            ) 
            (exists 
                (?bucket.n.01 - bucket.n.01) 
                (inside ?catsup.n.01_1 ?bucket.n.01)
            )
        )
    )
)