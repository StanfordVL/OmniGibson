(define (problem packing_picnics_0)
    (:domain igibson)

    (:objects
     	box.n.01_1 box.n.01_2 box.n.01_3 - box.n.01
    	floor.n.01_1 - floor.n.01
    	chip.n.04_1 chip.n.04_2 - chip.n.04
    	cabinet.n.01_1 - cabinet.n.01
    	sandwich.n.01_1 sandwich.n.01_2 sandwich.n.01_3 sandwich.n.01_4 - sandwich.n.01
    	countertop.n.01_1 - countertop.n.01
    	melon.n.01_1 - melon.n.01
    	strawberry.n.01_1 strawberry.n.01_2 strawberry.n.01_3 strawberry.n.01_4 - strawberry.n.01
    	electric_refrigerator.n.01_1 - electric_refrigerator.n.01
    	grape.n.01_1 grape.n.01_2 grape.n.01_3 grape.n.01_4 - grape.n.01
    	peach.n.03_1 peach.n.03_2 - peach.n.03
    	pop.n.02_1 pop.n.02_2 - pop.n.02
    	beer.n.01_1 beer.n.01_2 - beer.n.01
    	water.n.06_1 water.n.06_2 water.n.06_3 water.n.06_4 - water.n.06
    	sink.n.01_1 - sink.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor box.n.01_1 floor.n.01_1) 
        (onfloor box.n.01_2 floor.n.01_1) 
        (onfloor box.n.01_3 floor.n.01_1) 
        (inside chip.n.04_1 cabinet.n.01_1) 
        (inside chip.n.04_2 cabinet.n.01_1) 
        (ontop sandwich.n.01_1 countertop.n.01_1) 
        (ontop sandwich.n.01_2 countertop.n.01_1) 
        (ontop sandwich.n.01_3 countertop.n.01_1) 
        (ontop sandwich.n.01_4 countertop.n.01_1) 
        (ontop melon.n.01_1 countertop.n.01_1) 
        (inside strawberry.n.01_1 electric_refrigerator.n.01_1) 
        (inside strawberry.n.01_2 electric_refrigerator.n.01_1) 
        (inside strawberry.n.01_3 electric_refrigerator.n.01_1) 
        (inside strawberry.n.01_4 electric_refrigerator.n.01_1) 
        (inside grape.n.01_1 electric_refrigerator.n.01_1) 
        (inside grape.n.01_2 electric_refrigerator.n.01_1) 
        (inside grape.n.01_3 electric_refrigerator.n.01_1) 
        (inside grape.n.01_4 electric_refrigerator.n.01_1) 
        (inside peach.n.03_1 electric_refrigerator.n.01_1) 
        (inside peach.n.03_2 electric_refrigerator.n.01_1) 
        (inside pop.n.02_1 electric_refrigerator.n.01_1) 
        (inside pop.n.02_2 electric_refrigerator.n.01_1) 
        (inside beer.n.01_1 electric_refrigerator.n.01_1) 
        (inside beer.n.01_2 electric_refrigerator.n.01_1) 
        (inside water.n.06_1 cabinet.n.01_1) 
        (inside water.n.06_2 cabinet.n.01_1) 
        (inside water.n.06_3 cabinet.n.01_1) 
        (inside water.n.06_4 cabinet.n.01_1) 
        (inroom floor.n.01_1 kitchen) 
        (inroom electric_refrigerator.n.01_1 kitchen) 
        (inroom countertop.n.01_1 kitchen) 
        (inroom cabinet.n.01_1 kitchen) 
        (inroom sink.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (exists 
                (?box.n.01 - box.n.01) 
                (and 
                    (forall 
                        (?chip.n.04 - chip.n.04) 
                        (inside ?chip.n.04 ?box.n.01)
                    ) 
                    (forall 
                        (?sandwich.n.01 - sandwich.n.01) 
                        (inside ?sandwich.n.01 ?box.n.01)
                    )
                )
            ) 
            (exists 
                (?box.n.01 - box.n.01) 
                (and 
                    (forall 
                        (?strawberry.n.01 - strawberry.n.01) 
                        (inside ?strawberry.n.01 ?box.n.01)
                    ) 
                    (forall 
                        (?grape.n.01 - grape.n.01) 
                        (inside ?grape.n.01 ?box.n.01)
                    ) 
                    (forall 
                        (?peach.n.03 - peach.n.03) 
                        (inside ?peach.n.03 ?box.n.01)
                    ) 
                    (nextto ?melon.n.01_1 ?box.n.01)
                )
            ) 
            (exists 
                (?box.n.01 - box.n.01) 
                (and 
                    (forall 
                        (?pop.n.02 - pop.n.02) 
                        (inside ?pop.n.02 ?box.n.01)
                    ) 
                    (forall 
                        (?beer.n.01 - beer.n.01) 
                        (inside ?beer.n.01 ?box.n.01)
                    ) 
                    (forall 
                        (?water.n.06 - water.n.06) 
                        (inside ?water.n.06 ?box.n.01)
                    )
                )
            ) 
            (forall 
                (?box.n.01 - box.n.01) 
                (or 
                    (onfloor ?box.n.01 ?floor.n.01_1) 
                    (ontop ?box.n.01 ?countertop.n.01_1)
                )
            )
        )
    )
)