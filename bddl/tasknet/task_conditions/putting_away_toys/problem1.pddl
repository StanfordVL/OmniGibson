(define (problem putting_away_toys_1)
    (:domain igibson)

    (:objects
        drawing1 - drawing
        chest1 - chest
        plaything1 plaything10 plaything2 plaything3 plaything4 plaything5 plaything6 plaything7 plaything8 plaything9 - plaything
        coffee_table1 - coffee_table
        sofa_chair1 - sofa_chair
        sofa1 - sofa
        floor_lamp1 - floor_lamp
        plush1 plush2 plush3 - plush
        book1 - book
    )
    
    (:init 
        (nextto drawing1 chest1) 
        (ontop plaything1 coffee_table1) 
        (under plaything2 coffee_table1) 
        (under plaything3 sofa_chair1) 
        (ontop plaything4 sofa1) 
        (under plaything5 sofa_chair1) 
        (under plaything6 sofa_chair1) 
        (ontop plaything7 sofa1) 
        (under plaything8 coffee_table1) 
        (nextto plaything9 floor_lamp1) 
        (inside plaything10 chest1) 
        (nextto plush1 chest1) 
        (ontop plush2 sofa_chair1) 
        (ontop plush3 sofa_chair1) 
        (ontop book1 coffee_table1) 
        (inroom chest1 living room) 
        (inroom sofa1 living room) 
        (inroom coffee_table1 living room) 
        (inroom sofa_chair1 living room) 
        (inroom floor_lamp1 living room)
    )
    
    (:goal 
        (and 
            (and 
                (forn 
                    (10) 
                    (?plaything - plaything) 
                    (inside ?plaything ?chest1)
                ) 
                (forn 
                    (3) 
                    (?plush - plush) 
                    (inside ?plush ?chest1)
                ) 
                (inside ?book ?chest1) 
                (inside ?drawing ?chest1)
            )
        )
    )
)