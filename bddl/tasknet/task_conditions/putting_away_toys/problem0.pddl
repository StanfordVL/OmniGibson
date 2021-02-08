(define (problem putting_away_toys_0)
    (:domain igibson)

    (:objects
        plaything1 plaything10 plaything11 plaything12 plaything2 plaything3 plaything4 plaything5 plaything6 plaything7 plaything8 plaything9 - plaything
        sofa1 - sofa
        tv1 - tv
        coffee_table1 - coffee_table
        shelf1 shelf2 - shelf
        drawing1 - drawing
        plush1 - plush
    )
    
    (:init 
        (ontop plaything1 sofa1) 
        (nextto plaything2 tv1) 
        (under plaything3 coffee_table1) 
        (under plaything4 sofa1) 
        (inside plaything5 sofa1) 
        (ontop plaything6 tv1) 
        (nextto plaything7 shelf2) 
        (under plaything8 sofa1) 
        (nextto plaything9 tv1) 
        (under plaything10 sofa1) 
        (nextto plaything11 shelf1) 
        (ontop plaything12 coffee_table1) 
        (under drawing1 sofa1) 
        (nextto plush1 tv1)
    )
    
    (:goal 
        (and 
            (exists 
                (?shelf - shelf) 
                (forall 
                    (?plaything - plaything) 
                    (inside ?plaything ?shelf)
                )
            ) 
            (exists 
                (?shelf - shelf) 
                (forall 
                    (?plush - plush) 
                    (inside ?plush ?shelf)
                )
            ) 
            (exists 
                (?shelf - shelf) 
                (forall 
                    (?drawing - drawing) 
                    (inside ?drawing ?shelf)
                )
            )
        )
    )
)