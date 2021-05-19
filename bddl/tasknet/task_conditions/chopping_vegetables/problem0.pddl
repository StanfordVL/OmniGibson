(define (problem chopping_vegetables_0)
    (:domain igibson)

    (:objects
     	vegetable.n.01_1 vegetable.n.01_2 vegetable.n.01_3 vegetable.n.01_4 vegetable.n.01_5 vegetable.n.01_6 - vegetable.n.01
    	countertop.n.01_1 - countertop.n.01
    	onion.n.03_1 onion.n.03_2 - onion.n.03
    	electric_refrigerator.n.01_1 - electric_refrigerator.n.01
    	knife.n.01_1 - knife.n.01
    	dish.n.02_1 dish.n.02_2 - dish.n.02
    	cabinet.n.01_1 - cabinet.n.01
    	sink.n.01_1 - sink.n.01
    	floor.n.01_1 - floor.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (ontop vegetable.n.01_1 countertop.n.01_1) 
        (ontop vegetable.n.01_2 countertop.n.01_1) 
        (ontop vegetable.n.01_3 countertop.n.01_1) 
        (ontop vegetable.n.01_4 countertop.n.01_1) 
        (ontop vegetable.n.01_5 countertop.n.01_1) 
        (ontop vegetable.n.01_6 countertop.n.01_1) 
        (inside onion.n.03_1 electric_refrigerator.n.01_1) 
        (inside onion.n.03_2 electric_refrigerator.n.01_1) 
        (ontop knife.n.01_1 countertop.n.01_1) 
        (inside dish.n.02_1 cabinet.n.01_1) 
        (inside dish.n.02_2 cabinet.n.01_1) 
        (inroom countertop.n.01_1 kitchen) 
        (inroom cabinet.n.01_1 kitchen) 
        (inroom sink.n.01_1 kitchen) 
        (inroom electric_refrigerator.n.01_1 kitchen) 
        (inroom floor.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (nextto ?knife.n.01_1 ?sink.n.01_1) 
            (exists 
                (?dish.n.02 - dish.n.02) 
                (forall 
                    (?vegetable.n.01 - vegetable.n.01) 
                    (inside ?vegetable.n.01 ?dish.n.02)
                )
            ) 
            (exists 
                (?dish.n.02 - dish.n.02) 
                (forall 
                    (?onion.n.03 - onion.n.03) 
                    (inside ?onion.n.03 ?dish.n.02)
                )
            ) 
            (forall 
                (?vegetable.n.01 - vegetable.n.01) 
                (sliced ?vegetable.n.01)
            ) 
            (forall 
                (?onion.n.03 - onion.n.03) 
                (sliced ?onion.n.03)
            )
        )
    )
)