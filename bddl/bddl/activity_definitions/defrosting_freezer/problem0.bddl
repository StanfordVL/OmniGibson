(define (problem defrosting_freezer_0)
    (:domain igibson)

    (:objects
     	floor.n.01_1 - floor.n.01
    	electric_refrigerator.n.01_1 - electric_refrigerator.n.01
    	sink.n.01_1 - sink.n.01
    	countertop.n.01_1 - countertop.n.01
    	receptacle.n.01_1 - receptacle.n.01
    	bucket.n.01_1 - bucket.n.01
    	scraper.n.01_1 - scraper.n.01
    	towel.n.01_1 - towel.n.01
    	rag.n.01_1 - rag.n.01
    	food.n.02_1 food.n.02_2 food.n.02_3 - food.n.02
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (not 
            (stained sink.n.01_1)
        ) 
        (onfloor receptacle.n.01_1 floor.n.01_1) 
        (onfloor bucket.n.01_1 floor.n.01_1) 
        (ontop scraper.n.01_1 countertop.n.01_1) 
        (ontop towel.n.01_1 countertop.n.01_1) 
        (not 
            (stained towel.n.01_1)
        ) 
        (ontop rag.n.01_1 countertop.n.01_1) 
        (not 
            (soaked rag.n.01_1)
        ) 
        (inside food.n.02_1 electric_refrigerator.n.01_1) 
        (inside food.n.02_2 electric_refrigerator.n.01_1) 
        (inside food.n.02_3 electric_refrigerator.n.01_1) 
        (inroom floor.n.01_1 kitchen) 
        (inroom electric_refrigerator.n.01_1 kitchen) 
        (inroom sink.n.01_1 kitchen) 
        (inroom countertop.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (nextto ?receptacle.n.01_1 ?electric_refrigerator.n.01_1) 
            (nextto ?bucket.n.01_1 ?countertop.n.01_1) 
            (ontop ?scraper.n.01_1 ?electric_refrigerator.n.01_1) 
            (ontop ?towel.n.01_1 ?countertop.n.01_1) 
            (inside ?rag.n.01_1 ?sink.n.01_1) 
            (soaked ?rag.n.01_1) 
            (forall 
                (?food.n.02 - food.n.02) 
                (inside ?food.n.02 ?bucket.n.01_1)
            )
        )
    )
)