(define (problem clearing_the_table_after_dinner_0)
    (:domain igibson)

    (:objects
     	floor.n.01_1 - floor.n.01
    	chair.n.01_1 chair.n.01_2 - chair.n.01
    	table.n.02_1 - table.n.02
    	cup.n.01_1 cup.n.01_2 - cup.n.01
    	bucket.n.01_1 bucket.n.01_2 - bucket.n.01
    	plate.n.04_1 plate.n.04_2 plate.n.04_3 plate.n.04_4 - plate.n.04
    	catsup.n.01_1 - catsup.n.01
    	crumb.n.03_1 crumb.n.03_2 crumb.n.03_3 crumb.n.03_4 crumb.n.03_5 - crumb.n.03
    	beverage.n.01_1 beverage.n.01_2 - beverage.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (ontop cup.n.01_1 table.n.02_1) 
        (ontop cup.n.01_2 table.n.02_1) 
        (onfloor bucket.n.01_1 floor.n.01_1) 
        (ontop plate.n.04_1 table.n.02_1) 
        (ontop plate.n.04_2 table.n.02_1) 
        (ontop plate.n.04_3 table.n.02_1) 
        (ontop plate.n.04_4 table.n.02_1) 
        (ontop catsup.n.01_1 table.n.02_1) 
        (ontop crumb.n.03_1 table.n.02_1) 
        (ontop crumb.n.03_2 table.n.02_1) 
        (onfloor crumb.n.03_3 floor.n.01_1) 
        (ontop crumb.n.03_4 chair.n.01_1) 
        (ontop crumb.n.03_5 chair.n.01_2) 
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
                (?crumb.n.03 - crumb.n.03) 
                (exists 
                    (?bucket.n.01 - bucket.n.01) 
                    (inside ?crumb.n.03 ?bucket.n.01)
                )
            ) 
            (forall 
                (?plate.n.04 - plate.n.04) 
                (exists 
                    (?bucket.n.01 - bucket.n.01) 
                    (inside ?plate.n.04 ?bucket.n.01)
                )
            ) 
            (exists 
                (?bucket.n.01 - bucket.n.01) 
                (inside ?catsup.n.01_1 ?bucket.n.01)
            )
        )
    )
)