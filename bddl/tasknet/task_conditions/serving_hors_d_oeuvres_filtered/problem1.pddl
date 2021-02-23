(define (problem serving_hors_d_oeuvres_1)
    (:domain igibson)

    (:objects
        tray.n.01_1 tray.n.01_2 - tray.n.01
        countertop.n.01_1 - countertop.n.01
        casserole.n.02_1 - casserole.n.02
        oven.n.01_1 - oven.n.01
        sausage.n.01_1 sausage.n.01_2 - sausage.n.01
        cherry.n.03_1 cherry.n.03_2 - cherry.n.03
        electric_refrigerator.n.01_1 - electric_refrigerator.n.01
        microwave.n.02_1 - microwave.n.02
        table.n.02_1 - table.n.02
    )
    
    (:init 
        (ontop tray.n.01_1 countertop.n.01_1)
        (ontop tray.n.01_2 countertop.n.01_1)
        (inside casserole.n.02_1 oven.n.01_1)
        (inside sausage.n.01_1 oven.n.01_1)
        (inside sausage.n.01_2 oven.n.01_1)
        (inside cherry.n.03_1 electric_refrigerator.n.01_1)
        (inside cherry.n.03_2 electric_refrigerator.n.01_1)
        (inroom microwave.n.02_1 kitchen)
        (inroom oven.n.01_1 kitchen)
        (inroom electric_refrigerator.n.01_1 kitchen)
        (inroom countertop.n.01_1 kitchen)
        (inroom table.n.02_1 dining_room)
    )
    
    (:goal
        (and
            (exists
                (?tray.n.01 - tray.n.01)
                (and
                    (forall
                        (?sausage.n.01 - sausage.n.01)
                        (ontop ?sausage.n.01 ?tray.n.01)
                    )
                    (forall
                        (?cherry.n.03 - cherry.n.03)
                        (not
                            (ontop ?cherry.n.03 ?tray.n.01)
                        )
                    )
                )
            )
            (exists
                (?tray.n.01 - tray.n.01)
                (and
                    (forall
                        (?cherry.n.03 - cherry.n.03)
                        (ontop ?cherry.n.03 ?tray.n.01)
                    )
                    (forall
                        (?sausage.n.01 - sausage.n.01)
                        (not
                            (ontop ?sausage.n.01 ?tray.n.01)
                        )
                    )
                )
            )
        )
    )
)
