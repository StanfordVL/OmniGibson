(define (problem serving_hors_d_oeuvres_1)
    (:domain igibson)

    (:objects
        tray1 tray2 - tray
        counter1 - counter
        casserole1 - casserole
        oven1 - oven
        sausage1 sausage10 sausage11 sausage12 sausage2 sausage3 sausage4 sausage5 sausage6 sausage7 sausage8 sausage9 - sausage
        cherry1 cherry10 cherry11 cherry12 cherry2 cherry3 cherry4 cherry5 cherry6 cherry7 cherry8 cherry9 - cherry
        fridge1 - fridge
        microwave1 - microwave
        table1 - table
    )
    
    (:init 
        (ontop tray1 counter1) 
        (ontop tray2 counter1) 
        (inside casserole1 oven1) 
        (inside sausage1 oven1) 
        (inside sausage2 oven1) 
        (inside sausage3 oven1) 
        (inside sausage4 oven1) 
        (inside sausage5 oven1) 
        (inside sausage6 oven1) 
        (inside sausage7 oven1) 
        (inside sausage8 oven1) 
        (inside sausage9 oven1) 
        (inside sausage10 oven1) 
        (inside sausage11 oven1) 
        (inside sausage12 oven1) 
        (inside sausage1 casserole1) 
        (inside sausage2 casserole1) 
        (inside sausage3 casserole1) 
        (inside sausage4 casserole1) 
        (inside sausage5 casserole1) 
        (inside sausage6 casserole1) 
        (inside sausage7 casserole1) 
        (inside sausage8 casserole1) 
        (inside sausage9 casserole1) 
        (inside sausage10 casserole1) 
        (inside sausage11 casserole1) 
        (inside sausage12 casserole1) 
        (inside cherry1 fridge1) 
        (inside cherry2 fridge1) 
        (inside cherry3 fridge1) 
        (inside cherry4 fridge1) 
        (inside cherry5 fridge1) 
        (inside cherry6 fridge1) 
        (inside cherry7 fridge1) 
        (inside cherry8 fridge1) 
        (inside cherry9 fridge1) 
        (inside cherry10 fridge1) 
        (inside cherry11 fridge1) 
        (inside cherry12 fridge1) 
        (inroom microwave1 kitchen) 
        (inroom oven1 kitchen) 
        (inroom fridge1 kitchen) 
        (inroom counter1 kitchen) 
        (inroom table1 dining_room)
    )
    
    (:goal 
        (and 
            (exists 
                (?tray - tray) 
                (and 
                    (forall 
                        (?sausage - sausage) 
                        (ontop ?sausage ?tray)
                    ) 
                    (forall 
                        (?cherry - cherry) 
                        (not 
                            (ontop ?cherry ?tray)
                        )
                    )
                )
            ) 
            (exists 
                (?tray - tray) 
                (and 
                    (forall 
                        (?cherry - cherry) 
                        (ontop ?cherry ?tray)
                    ) 
                    (forall 
                        (?sausage - sausage) 
                        (not 
                            (ontop ?sausage ?tray)
                        )
                    )
                )
            )
        )
    )
)
