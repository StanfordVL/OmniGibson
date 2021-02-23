(define (problem serving_hors_d_oeuvres_0)
    (:domain igibson)

    (:objects
        chip.n.04_1 chip.n.04_2 - chip.n.04
        electric_refrigerator.n.01_1 - electric_refrigerator.n.01
        tray.n.01_1 - tray.n.01
        countertop.n.01_1 - countertop.n.01
        plate.n.04_1 plate.n.04_2  - plate.n.04
        cabinet.n.01_1 - cabinet.n.01
        table.n.02_1 - table.n.02
    )
    
    (:init 
        (inside chip.n.04_1 electric_refrigerator.n.01_1)
        (inside chip.n.04_2 electric_refrigerator.n.01_1)
        (ontop tray.n.01_1 countertop.n.01_1)
        (inside plate.n.04_1 cabinet.n.01_1)
        (inside plate.n.04_2 cabinet.n.01_1)
        (inroom countertop.n.01_1 kitchen)
        (inroom electric_refrigerator.n.01_1 kitchen)
        (inroom table.n.02_1 living_room)
        (inroom cabinet.n.01_1 kitchen)
    )
    
    (:goal 
        (and 
            (forall 
                (?tray.n.01 - tray.n.01)
                (fornpairs
                    (2)
                    (?chip.n.04 - chip.n.04)
                    (?plate.n.04 - plate.n.04)
                    (and 
                        (ontop ?chip.n.04 ?plate.n.04)
                        (ontop ?plate.n.04 ?tray.n.01)
                    )
                )
            ) 
            (forpairs
                (?tray.n.01 - tray.n.01)
                (?table.n.02 - table.n.02)
                (ontop ?tray.n.01 ?table.n.02)
            )
        )
    )
)
