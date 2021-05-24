(define (problem preparing_salad_0)
    (:domain igibson)

    (:objects
    	electric_refrigerator.n.01_1 - electric_refrigerator.n.01
    	lettuce.n.03_1 lettuce.n.03_2 - lettuce.n.03
    	countertop.n.01_1 - countertop.n.01
    	apple.n.01_1 apple.n.01_2 - apple.n.01
    	tomato.n.01_1 tomato.n.01_2 - tomato.n.01
    	radish.n.01_1 radish.n.01_2 - radish.n.01
        carving_knife.n.01_1 - carving_knife.n.01
    	bowl.n.01_1 bowl.n.01_2 - bowl.n.01
    	cabinet.n.01_1 - cabinet.n.01
    	sink.n.01_1 - sink.n.01
    	floor.n.01_1 - floor.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (ontop lettuce.n.03_1 countertop.n.01_1) 
        (ontop lettuce.n.03_2 countertop.n.01_1) 
        (ontop apple.n.01_1 countertop.n.01_1) 
        (ontop apple.n.01_2 countertop.n.01_1) 
        (inside tomato.n.01_1 electric_refrigerator.n.01_1) 
        (inside tomato.n.01_2 electric_refrigerator.n.01_1) 
        (ontop radish.n.01_1 countertop.n.01_1) 
        (ontop radish.n.01_2 countertop.n.01_1) 
        (inside bowl.n.01_1 cabinet.n.01_1) 
        (not 
            (dusty bowl.n.01_1)
        ) 
        (inside bowl.n.01_2 cabinet.n.01_1) 
        (not 
            (dusty bowl.n.01_2)
        ) 
        (inside carving_knife.n.01_1 cabinet.n.01_1)
        (inroom sink.n.01_1 kitchen) 
        (inroom countertop.n.01_1 kitchen) 
        (inroom electric_refrigerator.n.01_1 kitchen) 
        (inroom cabinet.n.01_1 kitchen) 
        (inroom floor.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forpairs 
                (?lettuce.n.03 - lettuce.n.03) 
                (?bowl.n.01 - bowl.n.01) 
                (inside ?lettuce.n.03 ?bowl.n.01)
            ) 
            (forpairs 
                (?apple.n.01 - apple.n.01) 
                (?bowl.n.01 - bowl.n.01) 
                (and 
                    (sliced ?apple.n.01) 
                    (inside ?apple.n.01 ?bowl.n.01)
                )
            ) 
            (forpairs 
                (?tomato.n.01 - tomato.n.01) 
                (?bowl.n.01 - bowl.n.01) 
                (and 
                    (inside ?tomato.n.01 ?bowl.n.01) 
                    (sliced ?tomato.n.01)
                )
            ) 
            (forpairs 
                (?radish.n.01 - radish.n.01) 
                (?bowl.n.01 - bowl.n.01) 
                (inside ?radish.n.01 ?bowl.n.01)
            )
        )
    )
)
