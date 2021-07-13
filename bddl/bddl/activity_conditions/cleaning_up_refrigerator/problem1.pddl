(define (problem cleaning_up_refrigerator_1)
    (:domain igibson)

    (:objects
     	rag.n.01_1 - rag.n.01
    	countertop.n.01_1 - countertop.n.01
    	soap.n.01_1 - soap.n.01
    	bowl.n.01_1 bowl.n.01_2 bowl.n.01_3 - bowl.n.01
    	electric_refrigerator.n.01_1 - electric_refrigerator.n.01
    	olive.n.04_1 olive.n.04_2 - olive.n.04
    	vegetable_oil.n.01_1 - vegetable_oil.n.01
    	sink.n.01_1 - sink.n.01
    	floor.n.01_1 - floor.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (ontop rag.n.01_1 countertop.n.01_1) 
        (soaked rag.n.01_1) 
        (ontop soap.n.01_1 countertop.n.01_1) 
        (inside bowl.n.01_1 electric_refrigerator.n.01_1) 
        (inside bowl.n.01_2 electric_refrigerator.n.01_1) 
        (inside bowl.n.01_3 electric_refrigerator.n.01_1) 
        (inside olive.n.04_1 electric_refrigerator.n.01_1) 
        (inside olive.n.04_2 electric_refrigerator.n.01_1) 
        (inside vegetable_oil.n.01_1 electric_refrigerator.n.01_1) 
        (stained electric_refrigerator.n.01_1) 
        (dusty electric_refrigerator.n.01_1) 
        (inroom sink.n.01_1 kitchen) 
        (inroom countertop.n.01_1 kitchen) 
        (inroom electric_refrigerator.n.01_1 kitchen) 
        (inroom floor.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forall 
                (?bowl.n.01 - bowl.n.01) 
                (nextto ?bowl.n.01 ?sink.n.01_1)
            ) 
            (forall 
                (?olive.n.04 - olive.n.04) 
                (touching ?olive.n.04 ?countertop.n.01_1)
            ) 
            (ontop ?vegetable_oil.n.01_1 ?countertop.n.01_1) 
            (not 
                (stained ?electric_refrigerator.n.01_1)
            ) 
            (not 
                (dusty ?electric_refrigerator.n.01_1)
            )
        )
    )
)