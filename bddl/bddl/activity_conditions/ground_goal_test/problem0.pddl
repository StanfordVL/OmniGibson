(define (problem serving_hors_d_oeuvres_1)
    (:domain igibson)

    (:objects
        tray.n.01_1 tray.n.01_2 - tray.n.01
        countertop.n.01_1 - countertop.n.01
        oven.n.01_1 - oven.n.01
        sausage.n.01_1 sausage.n.01_2 - sausage.n.01
        cherry.n.03_1 cherry.n.03_2 - cherry.n.03
        electric_refrigerator.n.01_1 - electric_refrigerator.n.01
    )
    
    (:init 
        (ontop tray.n.01_1 countertop.n.01_1)
        (ontop tray.n.01_2 countertop.n.01_1)
        (inside sausage.n.01_1 oven.n.01_1)
        (inside sausage.n.01_2 oven.n.01_1)
        (inside cherry.n.03_1 electric_refrigerator.n.01_1)
        (inside cherry.n.03_2 electric_refrigerator.n.01_1)
        (inroom oven.n.01_1 kitchen)
        (inroom electric_refrigerator.n.01_1 kitchen)
        (inroom countertop.n.01_1 kitchen)
    )

    (:goal
        (and
            (and
                (ontop ?cherry.n.03_1 ?tray.n.01_1)
                (ontop ?cherry.n.03_2 ?tray.n.01_1)
            )
            (or
                (ontop ?cherry.n.03_1 ?tray.n.01_1)
                (ontop ?cherry.n.03_2 ?tray.n.01_1)
            )
            (not
                (and
                    (ontop ?cherry.n.03_1 ?tray.n.01_1)
                    (ontop ?cherry.n.03_2 ?tray.n.01_1)
                )
            )
            (not
                (or
                    (ontop ?cherry.n.03_1 ?tray.n.01_1)
                    (ontop ?cherry.n.03_2 ?tray.n.01_1)
                )
            )
            (not
                (ontop ?cherry.n.03_1 ?tray.n.01_1)
            )
            (exists
                (?tray.n.01 - tray.n.01)
                (ontop ?sausage.n.01_1 ?tray.n.01)
            )
            (forall
                (?tray.n.01 - tray.n.01)
                (ontop ?sausage.n.01_1 ?tray.n.01)
            )
            (forn
                (1)
                (?cherry.n.03 - cherry.n.03)
                (nextto ?cherry.n.03 ?sausage.n.01_1)
            )
            (forpairs
                (?cherry.n.03 - cherry.n.03)
                (?tray.n.01 - tray.n.01)
                (inside ?cherry.n.03 ?tray.n.01)
            )
            (fornpairs
                (1)
                (?cherry.n.03 - cherry.n.03)
                (?tray.n.01 - tray.n.01)
                (under ?cherry.n.03 ?tray.n.01)
            )
            (fornpairs
                (2)
                (?cherry.n.03 - cherry.n.03)
                (?tray.n.01 - tray.n.01)
                (under ?cherry.n.03 ?tray.n.01)
            )
        )
    )
)
