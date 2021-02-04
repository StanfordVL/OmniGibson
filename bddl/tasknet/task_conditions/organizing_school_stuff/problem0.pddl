(define (problem organizing_school_stuff_0
    (:domain igibson)

    (:objects
        backpack1 - backpack
        sofa1 - sofa
        book1 book2 book3 book4 book5 - book
        sofa_chair1 - sofa_chair
        tablet1 - tablet
        pen1 pen2 - pen
        coffee_table1 - coffee_table
        shelf1 shelf2 - shelf
        pencil1 pencil2 - pencil
        folder1 folder2 - folder
    )
    
    (:init 
        (ontop backpack1 sofa1) 
        (ontop book1 sofa_chair1) 
        (ontop book2 sofa_chair1) 
        (nextto book3 sofa_chair1) 
        (nextto book4 sofa_chair1) 
        (ontop tablet1 sofa1) 
        (ontop pen1 coffee_table1) 
        (ontop pen2 shelf1) 
        (ontop pencil1 coffee_table1) 
        (under pencil2 coffee_table1) 
        (ontop folder1 shelf1) 
        (ontop folder2 shelf2) 
        (ontop book5 sofa1)
    )
    
    (:goal 
        (and 
            (ontop ?backpack1 ?sofa_chair1) 
            (forall 
                (?pen - pen) 
                (inside ?pen ?backpack1)
            ) 
            (forall 
                (?pencil - pencil) 
                (inside ?pencil ?backpack1)
            ) 
            (inside ?tablet1 ?backpack1) 
            (imply 
                (ontop ?folder1 ?shelf1) 
                (ontop ?folder2 ?shelf1)
            ) 
            (forn 
                (2) 
                (?book - book) 
                (inside ?book ?backpack1)
            ) 
            (forn 
                (3) 
                (?book - book) 
                (ontop ?book ?shelf2)
            )
        )
    )
)