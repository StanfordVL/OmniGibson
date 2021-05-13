(define (problem putting_leftovers_away_0)
    (:domain igibson)

    (:objects
     	pasta.n.02_1 pasta.n.02_2 pasta.n.02_3 pasta.n.02_4 - pasta.n.02
    	floor.n.01_1 - floor.n.01
    	sauce.n.01_1 sauce.n.01_2 sauce.n.01_3 sauce.n.01_4 - sauce.n.01
    	electric_refrigerator.n.01_1 - electric_refrigerator.n.01
    	stove.n.01_1 - stove.n.01
    	cabinet.n.01_1 cabinet.n.01_2 cabinet.n.01_3 - cabinet.n.01
    	dishwasher.n.01_1 - dishwasher.n.01
    	sink.n.01_1 - sink.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor pasta.n.02_1 floor.n.01_1) 
        (onfloor pasta.n.02_2 floor.n.01_1) 
        (onfloor pasta.n.02_3 floor.n.01_1) 
        (onfloor pasta.n.02_4 floor.n.01_1) 
        (onfloor sauce.n.01_1 floor.n.01_1) 
        (onfloor sauce.n.01_2 floor.n.01_1) 
        (inside sauce.n.01_3 electric_refrigerator.n.01_1) 
        (inside sauce.n.01_4 electric_refrigerator.n.01_1) 
        (inroom stove.n.01_1 kitchen) 
        (inroom electric_refrigerator.n.01_1 kitchen) 
        (inroom cabinet.n.01_1 kitchen) 
        (inroom cabinet.n.01_2 kitchen) 
        (inroom cabinet.n.01_3 kitchen) 
        (inroom dishwasher.n.01_1 kitchen) 
        (inroom sink.n.01_1 kitchen) 
        (inroom floor.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forall 
                (?pasta.n.02 - pasta.n.02) 
                (and 
                    (inside ?pasta.n.02 ?electric_refrigerator.n.01_1) 
                    (nextto ?pasta.n.02 ?pasta.n.02)
                )
            ) 
            (forall 
                (?sauce.n.01 - sauce.n.01) 
                (and 
                    (inside ?sauce.n.01 ?electric_refrigerator.n.01_1) 
                    (nextto ?sauce.n.01 ?sauce.n.01)
                )
            )
        )
    )
)