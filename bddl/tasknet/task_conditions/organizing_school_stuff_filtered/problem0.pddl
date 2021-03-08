(define 
    (problem organizing_school_stuff_0) 
    (:domain igibson)
    (:objects
        basket.n.01_1 - basket.n.01
        notebook.n.01_1 notebook.n.01_2 - notebook.n.01
        shelf.n.01_1 shelf.n.01_2 - shelf.n.01
        laptop.n.01_1 - laptop.n.01
        pen.n.01_1 - pen.n.01
        table.n.02_1 - table.n.02
        pencil.n.01_1 - pencil.n.01
        eraser.n.01_1 - eraser.n.01
    )
    (:init 
        (ontop basket.n.01_1 table.n.02_1)
        (ontop notebook.n.01_1 shelf.n.01_1) 
        (ontop notebook.n.01_2 shelf.n.01_2)  
        (ontop laptop.n.01_1 table.n.02_1) 
        (ontop pen.n.01_1 shelf.n.01_1) 
        (ontop pencil.n.01_1 shelf.n.01_2) 
        (ontop eraser.n.01_1 shelf.n.01_2) 
        (inroom shelf.n.01_1 living_room)
        (inroom shelf.n.01_2 living_room)
        (inroom table.n.02_1 living_room)
    )
    
    (:goal 
        (and 
            (ontop ?basket.n.01_1 ?table.n.02_1) 
            (forall 
                (?pen.n.01 - pen.n.01) 
                (inside ?pen.n.01 ?basket.n.01_1)
            ) 
            (forall 
                (?pencil.n.01 - pencil.n.01) 
                (inside ?pencil.n.01 ?basket.n.01_1)
            ) 
            (inside ?laptop.n.01_1 ?basket.n.01_1) 
            (imply 
                (ontop ?eraser.n.01_1 ?shelf.n.01_1) 
                (ontop ?eraser.n.01_2 ?shelf.n.01_1)
            ) 
            (forn 
                (2) 
                (?notebook.n.01 - notebook.n.01) 
                (inside ?notebook.n.01 ?basket.n.01_1)
            ) 
            (forn 
                (3) 
                (?notebook.n.01 - notebook.n.01) 
                (ontop ?notebook.n.01 ?shelf.n.01_2)
            )
        )
    )
)
