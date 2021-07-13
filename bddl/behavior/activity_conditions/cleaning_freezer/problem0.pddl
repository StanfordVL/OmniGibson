(define (problem cleaning_freezer_0)
    (:domain igibson)

    (:objects
     	electric_refrigerator.n.01_1 - electric_refrigerator.n.01
    	food.n.01_1 food.n.01_2 food.n.01_3 - food.n.01
    	cleansing_agent.n.01_1 - cleansing_agent.n.01
    	table.n.02_1 - table.n.02
    	towel.n.01_1 - towel.n.01
    	floor.n.01_1 - floor.n.01
    	sink.n.01_1 - sink.n.01
    	countertop.n.01_1 - countertop.n.01
    	stove.n.01_1 - stove.n.01
    	door.n.01_1 - door.n.01
    	chair.n.01_1 - chair.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (stained electric_refrigerator.n.01_1) 
        (inside food.n.01_1 electric_refrigerator.n.01_1) 
        (inside food.n.01_2 electric_refrigerator.n.01_1) 
        (inside food.n.01_3 electric_refrigerator.n.01_1) 
        (ontop cleansing_agent.n.01_1 table.n.02_1) 
        (ontop towel.n.01_1 table.n.02_1) 
        (inroom floor.n.01_1 kitchen) 
        (inroom sink.n.01_1 kitchen) 
        (inroom countertop.n.01_1 kitchen) 
        (inroom stove.n.01_1 kitchen) 
        (inroom electric_refrigerator.n.01_1 kitchen) 
        (inroom door.n.01_1 kitchen) 
        (inroom table.n.02_1 kitchen) 
        (inroom chair.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (not 
                (stained ?electric_refrigerator.n.01_1)
            ) 
            (forall 
                (?food.n.01 - food.n.01) 
                (not 
                    (inside ?food.n.01 ?electric_refrigerator.n.01_1)
                )
            )
        )
    )
)