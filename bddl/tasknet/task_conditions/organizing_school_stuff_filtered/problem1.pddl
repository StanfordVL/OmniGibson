(define (problem organizing_school_stuff_1) 
    (:domain igibson)

    (:objects
        ; laptop.n.01_1 - laptop.n.01
    	cabinet.n.01_1 cabinet.n.01_2 - cabinet.n.01
    	bag.n.01_1 - bag.n.01
        basket.n.01_1 - basket.n.01
        ; notebook.n.01_1 notebook.n.01_2 notebook.n.01_3 - notebook.n.01
    	notebook.n.01_1 - notebook.n.01
        ; eraser.n.01_1 eraser.n.01_2 - eraser.n.01
        eraser.n.01_1 - eraser.n.01
    	; pen.n.01_1 pen.n.01_2 pen.n.01_3 pen.n.01_4 - pen.n.01
    	pen.n.01_1 - pen.n.01
    	; pencil.n.01_1 pencil.n.01_2 - pencil.n.01
    	pencil.n.01_1 - pencil.n.01
    )
    
    (:init 
        ; (ontop laptop.n.01_1 cabinet.n.01_1) 
        (ontop bag.n.01_1 cabinet.n.01_1)
        (ontop basket.n.01_1 cabinet.n.01_2)
        (ontop notebook.n.01_1 cabinet.n.01_2)
        ; (ontop notebook.n.01_2 cabinet.n.01_1) 
        ; (ontop notebook.n.01_3 cabinet.n.01_1) 
        (inside eraser.n.01_1 cabinet.n.01_1)
        ; (ontop eraser.n.01_2 cabinet.n.01_1)
        (ontop pen.n.01_1 cabinet.n.01_1) 
        ; (ontop pen.n.01_2 cabinet.n.01_2) 
        ; (ontop pen.n.01_3 cabinet.n.01_2) 
        ; (ontop pen.n.01_4 cabinet.n.01_2) 
        (ontop pencil.n.01_1 cabinet.n.01_2) 
        ; (ontop pencil.n.01_2 cabinet.n.01_2)
        (inroom cabinet.n.01_1 bedroom)
        (inroom cabinet.n.01_2 bedroom)
    )
    
    (:goal 
        (and 
            (forall 
                (?notebook.n.01 - notebook.n.01) 
                (inside ?notebook.n.01 ?basket.n.01_1)
            )
            ; (forall
            ;     (?eraser.n.01 - eraser.n.01)
            ;     (inside ?eraser.n.01 ?basket.n.01_1)
            ; )
            ; (and 
            ;     (and 
            ;         (forall 
            ;             (?pen.n.01 - pen.n.01) 
            ;             (inside ?pen.n.01 ?bag.n.01_1)
            ;         ) 
            ;         (forall 
            ;             (?pencil.n.01 - pencil.n.01) 
            ;             (inside ?pencil.n.01 ?bag.n.01_1)
            ;         )
            ;     ) 
            ;     (not 
            ;         (open ?bag.n.01_1)
            ;     ) 
            ;     (inside ?bag.n.01_1 ?basket.n.01_1)
            ; ) 
            ; (inside ?laptop.n.01_1 ?basket.n.01_1)
            ; (not 
            ;     (open ?basket.n.01_1)
            ; )
        )
    )
)
