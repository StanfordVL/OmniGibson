(define (problem chopping_vegetables_0)
    (:domain igibson)

    (:objects
        tomato.n.01_1 tomato.n.01_2 - tomato.n.01
        mushroom.n.05_1 mushroom.n.05_2 - mushroom.n.05
        chestnut.n.03_1 chestnut.n.03_2 - chestnut.n.03
        countertop.n.01_1 - countertop.n.01
        vidalia_onion.n.01_1 vidalia_onion.n.01_2 - vidalia_onion.n.01
        electric_refrigerator.n.01_1 - electric_refrigerator.n.01
        knife.n.01_1 - knife.n.01
        dish.n.01_1 dish.n.01_2 - dish.n.01
        cabinet.n.01_1 - cabinet.n.01
        sink.n.01_1 - sink.n.01
        floor.n.01_1 - floor.n.01
        agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (ontop tomato.n.01_1 countertop.n.01_1) 
        (ontop tomato.n.01_2 countertop.n.01_1) 
        (ontop mushroom.n.05_1 countertop.n.01_1) 
        (ontop mushroom.n.05_2 countertop.n.01_1) 
        (ontop chestnut.n.03_1 countertop.n.01_1) 
        (ontop chestnut.n.03_2 countertop.n.01_1) 
        (inside vidalia_onion.n.01_1 electric_refrigerator.n.01_1) 
        (inside vidalia_onion.n.01_2 electric_refrigerator.n.01_1) 
        (ontop knife.n.01_1 countertop.n.01_1) 
        (inside dish.n.01_1 cabinet.n.01_1) 
        (inside dish.n.01_2 cabinet.n.01_1) 
        (inroom countertop.n.01_1 kitchen) 
        (inroom cabinet.n.01_1 kitchen) 
        (inroom sink.n.01_1 kitchen) 
        (inroom electric_refrigerator.n.01_1 kitchen) 
        (inroom floor.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forall 
                (?tomato.n.01 - tomato.n.01) 
                (and 
                    (exists 
                        (?dish.n.01 - dish.n.01) 
                        (inside ?tomato.n.01 ?dish.n.01)
                    ) 
                    (sliced ?tomato.n.01)
                )
            ) 
            (forall 
                (?mushroom.n.05 - mushroom.n.05) 
                (and 
                    (exists 
                        (?dish.n.01 - dish.n.01) 
                        (inside ?mushroom.n.05 ?dish.n.01)
                    ) 
                    (sliced ?mushroom.n.05)
                )
            ) 
            (forall 
                (?chestnut.n.03 - chestnut.n.03) 
                (and 
                    (exists 
                        (?dish.n.01 - dish.n.01) 
                        (inside ?chestnut.n.03 ?dish.n.01)
                    ) 
                    (sliced ?chestnut.n.03)
                )
            ) 
            (forall 
                (?vidalia_onion.n.01 - vidalia_onion.n.01) 
                (and 
                    (exists 
                        (?dish.n.01 - dish.n.01) 
                        (inside ?vidalia_onion.n.01 ?dish.n.01)
                    ) 
                    (sliced ?vidalia_onion.n.01)
                )
            )
        )
    )
)