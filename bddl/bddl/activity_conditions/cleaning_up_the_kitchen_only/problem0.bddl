(define (problem cleaning_up_the_kitchen_only_0)
    (:domain igibson)

    (:objects
        bin.n.01_1 - bin.n.01
        floor.n.01_1 - floor.n.01
        soap.n.01_1 - soap.n.01
        cabinet.n.01_1 cabinet.n.01_2 - cabinet.n.01
        electric_refrigerator.n.01_1 - electric_refrigerator.n.01
        rag.n.01_1 - rag.n.01
        dustpan.n.02_1 - dustpan.n.02
        broom.n.01_1 - broom.n.01
        blender.n.01_1 - blender.n.01
        sink.n.01_1 - sink.n.01
        casserole.n.02_1 - casserole.n.02
        plate.n.04_1 - plate.n.04
        vegetable_oil.n.01_1 - vegetable_oil.n.01
        apple.n.01_1 - apple.n.01
        window.n.01_1 - window.n.01
        countertop.n.01_1 - countertop.n.01
        agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor bin.n.01_1 floor.n.01_1) 
        (inside soap.n.01_1 cabinet.n.01_1) 
        (inside rag.n.01_1 cabinet.n.01_1) 
        (not 
            (soaked rag.n.01_1)
        ) 
        (inside dustpan.n.02_1 cabinet.n.01_1) 
        (dusty dustpan.n.02_1) 
        (onfloor broom.n.01_1 floor.n.01_1) 
        (dusty broom.n.01_1) 
        (onfloor blender.n.01_1 floor.n.01_1) 
        (stained blender.n.01_1) 
        (inside casserole.n.02_1 electric_refrigerator.n.01_1) 
        (inside plate.n.04_1 electric_refrigerator.n.01_1) 
        (stained plate.n.04_1) 
        (inside vegetable_oil.n.01_1 electric_refrigerator.n.01_1) 
        (inside apple.n.01_1 electric_refrigerator.n.01_1) 
        (dusty floor.n.01_1) 
        (dusty cabinet.n.01_1) 
        (dusty cabinet.n.01_2) 
        (inroom floor.n.01_1 kitchen) 
        (inroom cabinet.n.01_1 kitchen) 
        (inroom cabinet.n.01_2 kitchen) 
        (inroom electric_refrigerator.n.01_1 kitchen) 
        (inroom sink.n.01_1 kitchen) 
        (inroom window.n.01_1 kitchen) 
        (inroom countertop.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (ontop ?blender.n.01_1 ?countertop.n.01_1) 
            (nextto ?soap.n.01_1 ?sink.n.01_1) 
            (exists 
                (?cabinet.n.01 - cabinet.n.01) 
                (and 
                    (inside ?vegetable_oil.n.01_1 ?cabinet.n.01) 
                    (not 
                        (inside ?plate.n.04_1 ?cabinet.n.01)
                    )
                )
            ) 
            (exists 
                (?cabinet.n.01 - cabinet.n.01) 
                (and 
                    (inside ?plate.n.04_1 ?cabinet.n.01) 
                    (not 
                        (inside ?vegetable_oil.n.01_1 ?cabinet.n.01)
                    )
                )
            ) 
            (and 
                (not 
                    (dusty ?cabinet.n.01_1)
                ) 
                (not 
                    (dusty ?cabinet.n.01_2)
                ) 
                (not 
                    (dusty ?floor.n.01_1)
                )
            ) 
            (not 
                (stained ?plate.n.04_1)
            ) 
            (or 
                (nextto ?rag.n.01_1 ?sink.n.01_1) 
                (inside ?rag.n.01_1 ?sink.n.01_1)
            ) 
            (and 
                (inside ?casserole.n.02_1 ?electric_refrigerator.n.01_1) 
                (inside ?apple.n.01_1 ?electric_refrigerator.n.01_1)
            )
        )
    )
)