(define (problem filling_an_Easter_basket_1)
    (:domain igibson)

    (:objects
     	basket1 basket2 basket3 basket4 - basket
    	cabinet1 - cabinet
    	coloring_material1 coloring_material2 coloring_material3 coloring_material4 - coloring_material
    	shelf1 - shelf
    	egg1 egg2 egg3 egg4 egg5 egg6 egg7 egg8 - egg
    	counter1 - counter
    	chocolate1 chocolate2 chocolate3 chocolate4 - chocolate
    	crayon1 crayon2 crayon3 crayon4 crayon5 crayon6 crayon7 crayon8 - crayon
    )
    
    (:init 
        (inside basket1 cabinet1) 
        (inside basket2 cabinet1) 
        (inside basket3 cabinet1) 
        (inside basket4 cabinet1) 
        (ontop coloring_material1 shelf1) 
        (ontop coloring_material2 shelf1) 
        (ontop coloring_material3 shelf1) 
        (ontop coloring_material4 shelf1) 
        (not 
            (cooked egg1)
        ) 
        (not 
            (cooked egg2)
        ) 
        (not 
            (cooked egg3)
        ) 
        (not 
            (cooked egg4)
        ) 
        (not 
            (cooked egg5)
        ) 
        (not 
            (cooked egg6)
        ) 
        (not 
            (cooked egg7)
        ) 
        (not 
            (cooked egg8)
        ) 
        (and 
            (ontop egg1 counter1) 
            (ontop egg2 counter1) 
            (ontop egg3 counter1) 
            (ontop egg4 counter1) 
            (ontop egg5 counter1) 
            (ontop egg6 counter1) 
            (ontop egg7 counter1) 
            (ontop egg8 counter1)
        ) 
        (and 
            (ontop chocolate1 counter1) 
            (ontop chocolate2 counter1) 
            (ontop chocolate3 counter1) 
            (ontop chocolate4 counter1)
        ) 
        (and 
            (ontop crayon1 shelf1) 
            (ontop crayon2 shelf1) 
            (ontop crayon3 shelf1) 
            (ontop crayon4 shelf1) 
            (ontop crayon5 shelf1) 
            (ontop crayon6 shelf1) 
            (ontop crayon7 shelf1) 
            (ontop crayon8 shelf1)
        ) 
        (inroom cabinet1 kitchen) 
        (inroom shelf1 kitchen) 
        (inroom counter1 kitchen)
    )
    
    (:goal 
        (and 
            (forall 
                (?basket - basket) 
                (forn 
                    (2) 
                    (?egg - egg) 
                    (inside ?egg ?basket)
                )
            ) 
            (forpairs 
                (?chocolate - chocolate) 
                (?basket - basket) 
                (inside ?chocolate ?basket)
            ) 
            (forpairs 
                (?coloring_material - coloring_material) 
                (?basket - basket) 
                (inside ?coloring_material ?basket)
            ) 
            (forall 
                (?basket - basket) 
                (forn 
                    (2) 
                    (?crayon - crayon) 
                    (inside ?crayon ?basket)
                )
            )
        )
    )
)