(define (problem organizing_school_stuff_0)
    (:domain igibson)

    (:objects
        basket1 - basket
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
        (ontop basket1 sofa1) 
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
            (ontop ?basket1 ?sofa_chair1) 
            (forall 
                (?pen - pen) 
                (inside ?pen ?basket1)
            ) 
            (forall 
                (?pencil - pencil) 
                (inside ?pencil ?basket1)
            ) 
            (inside ?laptop1 ?basket1) 
            (imply 
                (ontop ?eraser1 ?shelf1) 
                (ontop ?eraser2 ?shelf1)
            ) 
            (forn 
                (2) 
                (?notebook - notebook) 
                (inside ?notebook ?basket1)
            ) 
            (forn 
                (3) 
                (?notebook - notebook) 
                (ontop ?notebook ?shelf2)
            )
        )
    )
)
