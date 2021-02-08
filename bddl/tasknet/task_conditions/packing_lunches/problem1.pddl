(define (problem packing_lunches_1)
    (:domain igibson)

    (:objects
     	box1 box2 - box
    	shelf1 - shelf
    	bag1 bag2 bag3 bag4 bag5 bag6 bag7 bag8 - bag
    	water1 water2 water3 water4 - water
    	bottle1 bottle2 bottle3 bottle4 - bottle
    	counter1 - counter
    	edible_fruit1 edible_fruit2 edible_fruit3 edible_fruit4 - edible_fruit
    	fridge1 - fridge
    	sandwich1 sandwich2 sandwich3 sandwich4 - sandwich
    	cookie1 cookie2 cookie3 cookie4 - cookie
    	napkin1 napkin2 napkin3 napkin4 - napkin
    	muffin1 muffin2 muffin3 muffin4 - muffin
    	pack1 pack2 pack3 pack4 - pack
    )
    
    (:init 
        (and 
            (ontop box1 shelf1) 
            (ontop box2 shelf1) 
            (and 
                (inside bag1 box1) 
                (under shelf1 bag1)
            ) 
            (and 
                (inside bag2 box1) 
                (under shelf1 bag2)
            ) 
            (and 
                (inside bag3 box1) 
                (under shelf1 bag3)
            ) 
            (and 
                (inside bag4 box1) 
                (under shelf1 bag4)
            ) 
            (and 
                (inside bag5 box1) 
                (under shelf1 bag5)
            ) 
            (and 
                (inside bag6 box1) 
                (under shelf1 bag6)
            ) 
            (and 
                (inside bag7 box1) 
                (under shelf1 bag7)
            ) 
            (and 
                (inside bag8 box1) 
                (under shelf1 bag8)
            )
        ) 
        (and 
            (and 
                (inside water1 bottle1) 
                (ontop bottle1 counter1) 
                (under counter1 water1)
            ) 
            (and 
                (inside water2 bottle2) 
                (ontop bottle2 counter1) 
                (under counter1 water2)
            ) 
            (and 
                (inside water3 bottle3) 
                (ontop bottle3 counter1) 
                (under counter1 water3)
            ) 
            (and 
                (inside water4 bottle4) 
                (ontop bottle4 counter1) 
                (under counter1 water4)
            )
        ) 
        (and 
            (inside edible_fruit1 fridge1) 
            (inside edible_fruit2 fridge1) 
            (inside edible_fruit3 fridge1) 
            (inside edible_fruit4 fridge1) 
            (inside sandwich1 fridge1) 
            (inside sandwich2 fridge1) 
            (inside sandwich3 fridge1) 
            (inside sandwich4 fridge1)
        ) 
        (and 
            (and 
                (inside cookie1 box2) 
                (under shelf1 cookie1)
            ) 
            (and 
                (inside cookie2 box2) 
                (under shelf1 cookie2)
            ) 
            (and 
                (inside cookie3 box2) 
                (under shelf1 cookie3)
            ) 
            (and 
                (inside cookie4 box2) 
                (under shelf1 cookie4)
            )
        ) 
        (and 
            (ontop napkin1 shelf1) 
            (ontop napkin2 shelf1) 
            (ontop napkin3 shelf1) 
            (ontop napkin4 shelf1) 
            (ontop muffin1 shelf1) 
            (ontop muffin2 shelf1) 
            (ontop muffin3 shelf1) 
            (ontop muffin4 shelf1)
        ) 
        (and 
            (ontop pack1 counter1) 
            (ontop pack2 counter1) 
            (ontop pack3 counter1) 
            (ontop pack4 counter1)
        ) 
        (inroom counter1 kitchen) 
        (inroom fridge1 kitchen) 
        (inroom shelf1 kitchen)
    )
    
    (:goal 
        (and 
            (forn 
                (4) 
                (?pack - pack) 
                (and 
                    (forpairs 
                        (?water - water) 
                        (?bottle - bottle) 
                        (and 
                            (inside ?water ?bottle) 
                            (inside ?bottle ?pack) 
                            (not 
                                (open ?bottle)
                            )
                        )
                    ) 
                    (fornpairs 
                        (4) 
                        (?bag - bag) 
                        (?sandwich - sandwich) 
                        (and 
                            (inside ?sandwich ?bag) 
                            (inside ?bag ?pack) 
                            (not 
                                (open ?bag)
                            )
                        )
                    ) 
                    (fornpairs 
                        (4) 
                        (?bag - bag) 
                        (?cookie - cookie) 
                        (and 
                            (inside ?cookie ?bag) 
                            (inside ?bag ?pack) 
                            (not 
                                (open ?bag)
                            )
                        )
                    )
                )
            ) 
            (fornpairs 
                (4) 
                (?pack - pack) 
                (?napkin - napkin) 
                (inside ?napkin ?pack)
            ) 
            (fornpairs 
                (4) 
                (?pack - pack) 
                (?edible_fruit - edible_fruit) 
                (inside ?edible_fruit ?pack)
            ) 
            (fornpairs 
                (4) 
                (?pack - pack) 
                (?muffin - muffin) 
                (inside ?muffin ?pack)
            ) 
            (forall 
                (?pack - pack) 
                (and 
                    (ontop ?pack ?counter1) 
                    (not 
                        (open ?pack)
                    )
                )
            )
        )
    )
)