(define (problem organizing_school_stuff_0)
    (:domain igibson)

    (:objects
        backpack1 - backpack
        sofa1 - sofa
        notebook1 notebook2 notebook3 notebook4 notebook5 - notebook
        sofa_chair1 - sofa_chair
        laptop1 - laptop
        pen1 pen2 - pen
        coffee_table1 - coffee_table
        shelf1 shelf2 - shelf
        pencil1 pencil2 - pencil
        eraser1 eraser2 - eraser
    )
    
    (:init 
        (ontop backpack1 sofa1) 
        (ontop notebook1 sofa_chair1) 
        (ontop notebook2 sofa_chair1) 
        (nextto notebook3 sofa_chair1) 
        (nextto notebook4 sofa_chair1) 
        (ontop laptop1 sofa1) 
        (ontop pen1 coffee_table1) 
        (ontop pen2 shelf1) 
        (ontop pencil1 coffee_table1) 
        (under pencil2 coffee_table1) 
        (ontop eraser1 shelf1) 
        (ontop eraser2 shelf2) 
        (ontop notebook5 sofa1)
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
            (inside ?laptop1 ?backpack1) 
            (imply 
                (ontop ?eraser1 ?shelf1) 
                (ontop ?eraser2 ?shelf1)
            ) 
            (forn 
                (2) 
                (?notebook - notebook) 
                (inside ?notebook ?backpack1)
            ) 
            (forn 
                (3) 
                (?notebook - notebook) 
                (ontop ?notebook ?shelf2)
            )
        )
    )
)
