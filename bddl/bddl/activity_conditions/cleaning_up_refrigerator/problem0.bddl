(define (problem cleaning_up_refrigerator_0)
    (:domain igibson)

    (:objects
     	rag.n.01_1 rag.n.01_2 - rag.n.01
    	cabinet.n.01_1 - cabinet.n.01
    	soap.n.01_1 - soap.n.01
    	countertop.n.01_1 - countertop.n.01
    	tray.n.01_1 tray.n.01_2 - tray.n.01
    	electric_refrigerator.n.01_1 - electric_refrigerator.n.01
    	bowl.n.01_1 - bowl.n.01
    	sink.n.01_1 - sink.n.01
            agent.n.01_1 - agent.n.01
            floor.n.01_1 - floor.n.01
    )
    
    (:init 
        (inside rag.n.01_1 cabinet.n.01_1) 
        (inside rag.n.01_2 cabinet.n.01_1) 
        (ontop soap.n.01_1 countertop.n.01_1) 
        (inside tray.n.01_1 electric_refrigerator.n.01_1) 
        (inside tray.n.01_2 electric_refrigerator.n.01_1) 
        (inside bowl.n.01_1 electric_refrigerator.n.01_1) 
        (not 
            (soaked rag.n.01_1)
        ) 
        (not 
            (soaked rag.n.01_2)
        ) 
        (stained tray.n.01_1) 
        (stained tray.n.01_2) 
        (dusty bowl.n.01_1) 
        (stained electric_refrigerator.n.01_1) 
        (inroom electric_refrigerator.n.01_1 kitchen) 
        (inroom countertop.n.01_1 kitchen) 
        (inroom sink.n.01_1 kitchen) 
        (inroom cabinet.n.01_1 kitchen) 
        (inroom floor.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forall 
                (?rag.n.01 - rag.n.01) 
                (nextto ?rag.n.01 ?sink.n.01_1)
            ) 
            (inside ?soap.n.01_1 ?sink.n.01_1) 
            (forall 
                (?tray.n.01 - tray.n.01) 
                (inside ?tray.n.01 ?electric_refrigerator.n.01_1)
            ) 
            (not 
                (stained ?tray.n.01_1)
            ) 
            (not 
                (stained ?tray.n.01_2)
            ) 
            (nextto ?bowl.n.01_1 ?sink.n.01_1) 
            (not 
                (dusty ?bowl.n.01_1)
            ) 
            (not 
                (stained ?electric_refrigerator.n.01_1)
            )
        )
    )
)