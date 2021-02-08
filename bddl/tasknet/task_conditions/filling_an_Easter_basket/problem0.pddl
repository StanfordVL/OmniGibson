(define (problem filling_an_Easter_basket_0)
    (:domain igibson)

    (:objects
     	basket1 basket2 basket3 basket4 - basket
    	shelf1 - shelf
    	chocolate1 chocolate2 chocolate3 chocolate4 chocolate5 chocolate6 chocolate7 chocolate8 - chocolate
    	candy1 candy2 candy3 candy4 candy5 candy6 candy7 candy8 - candy
    	egg1 egg2 egg3 egg4 egg5 egg6 egg7 egg8 - egg
    	foil1 foil2 foil3 foil4 - foil
    )
    
    (:init 
        (ontop basket1 shelf1) 
        (ontop basket2 shelf1) 
        (ontop basket3 shelf1) 
        (ontop basket4 shelf1) 
        (ontop chocolate1 shelf1) 
        (ontop chocolate2 shelf1) 
        (ontop chocolate3 shelf1) 
        (ontop chocolate4 shelf1) 
        (ontop chocolate5 shelf1) 
        (ontop chocolate6 shelf1) 
        (ontop chocolate7 shelf1) 
        (ontop chocolate8 shelf1) 
        (ontop candy1 shelf1) 
        (ontop candy2 shelf1) 
        (ontop candy3 shelf1) 
        (ontop candy4 shelf1) 
        (ontop candy5 shelf1) 
        (ontop candy6 shelf1) 
        (ontop candy7 shelf1) 
        (ontop candy8 shelf1) 
        (ontop egg1 shelf1) 
        (ontop egg2 shelf1) 
        (ontop egg3 shelf1) 
        (ontop egg4 shelf1) 
        (ontop egg5 shelf1) 
        (ontop egg6 shelf1) 
        (ontop egg7 shelf1) 
        (ontop egg8 shelf1) 
        (ontop foil1 shelf1) 
        (ontop foil2 shelf1) 
        (ontop foil3 shelf1) 
        (ontop foil4 shelf1)
    )
    
    (:goal 
        (and 
            (forn 
                (4) 
                (?basket - basket) 
                (and 
                    (forn 
                        (2) 
                        (?chocolate - chocolate) 
                        (inside ?chocolate ?basket)
                    ) 
                    (forn 
                        (2) 
                        (?candy - candy) 
                        (inside ?candy ?basket)
                    ) 
                    (forn 
                        (2) 
                        (?egg - egg) 
                        (inside ?egg ?basket)
                    ) 
                    (inside ?foil ?basket) 
                    (under ?foil ?chocolate) 
                    (under ?foil ?egg) 
                    (under ?foil ?candy)
                )
            ) 
            (scrubbed ?table1)
        )
    )
)