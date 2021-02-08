(define (problem serving_hors_d_oeuvres_1)
    (:domain igibson)

    (:objects
        platter1 platter2 - platter
        counter1 - counter
        dish1 - dish
        oven1 - oven
        meatball1 meatball10 meatball11 meatball12 meatball2 meatball3 meatball4 meatball5 meatball6 meatball7 meatball8 meatball9 - meatball
        sushi1 sushi10 sushi11 sushi12 sushi2 sushi3 sushi4 sushi5 sushi6 sushi7 sushi8 sushi9 - sushi
        fridge1 - fridge
        microwave1 - microwave
        table1 - table
    )
    
    (:init 
        (ontop platter1 counter1) 
        (ontop platter2 counter1) 
        (inside dish1 oven1) 
        (inside meatball1 oven1) 
        (inside meatball2 oven1) 
        (inside meatball3 oven1) 
        (inside meatball4 oven1) 
        (inside meatball5 oven1) 
        (inside meatball6 oven1) 
        (inside meatball7 oven1) 
        (inside meatball8 oven1) 
        (inside meatball9 oven1) 
        (inside meatball10 oven1) 
        (inside meatball11 oven1) 
        (inside meatball12 oven1) 
        (inside meatball1 dish1) 
        (inside meatball2 dish1) 
        (inside meatball3 dish1) 
        (inside meatball4 dish1) 
        (inside meatball5 dish1) 
        (inside meatball6 dish1) 
        (inside meatball7 dish1) 
        (inside meatball8 dish1) 
        (inside meatball9 dish1) 
        (inside meatball10 dish1) 
        (inside meatball11 dish1) 
        (inside meatball12 dish1) 
        (inside sushi1 fridge1) 
        (inside sushi2 fridge1) 
        (inside sushi3 fridge1) 
        (inside sushi4 fridge1) 
        (inside sushi5 fridge1) 
        (inside sushi6 fridge1) 
        (inside sushi7 fridge1) 
        (inside sushi8 fridge1) 
        (inside sushi9 fridge1) 
        (inside sushi10 fridge1) 
        (inside sushi11 fridge1) 
        (inside sushi12 fridge1) 
        (inroom microwave1 kitchen) 
        (inroom oven1 kitchen) 
        (inroom fridge1 kitchen) 
        (inroom counter1 kitchen) 
        (inroom table1 diningroom)
    )
    
    (:goal 
        (and 
            (exists 
                (?platter - platter) 
                (and 
                    (forall 
                        (?meatball - meatball) 
                        (ontop ?meatball ?platter)
                    ) 
                    (forall 
                        (?sushi - sushi) 
                        (not 
                            (ontop ?sushi ?platter)
                        )
                    )
                )
            ) 
            (exists 
                (?platter - platter) 
                (and 
                    (forall 
                        (?sushi - sushi) 
                        (ontop ?sushi ?platter)
                    ) 
                    (forall 
                        (?meatball - meatball) 
                        (not 
                            (ontop ?meatball ?platter)
                        )
                    )
                )
            )
        )
    )
)