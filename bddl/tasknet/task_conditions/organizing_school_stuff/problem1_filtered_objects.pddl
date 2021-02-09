(define (problem organizing_school_stuff_1)
    (:domain igibson)

    (:objects
     	laptop1 - laptop
    	shelf1 - shelf
    	bag1 - bag
    	table1 - table
    	basket1 - basket
    	notebook1 notebook2 notebook3 - notebook
    	eraser1 eraser2 - eraser
    	pen1 pen2 pen3 pen4 - pen
    	pencil1 pencil2 - pencil
    )
    
    (:init 
        (ontop laptop1 shelf1) 
        (and 
            (ontop bag1 table1) 
            (open bag1)
        ) 
        (and 
            (ontop basket1 table1) 
            (open basket1)
        ) 
        (and 
            (ontop notebook1 shelf1) 
            (ontop notebook2 shelf1) 
            (ontop notebook3 shelf1) 
            (ontop eraser1 shelf1) 
            (ontop eraser2 shelf1)
        ) 
        (and 
            (ontop pen1 table1) 
            (ontop pen2 table1) 
            (ontop pen3 table1) 
            (ontop pen4 table1) 
            (ontop pencil1 table1) 
            (ontop pencil2 table1)
        )
    )
    
    (:goal 
        (and 
            (and 
                (forall 
                    (?notebook - notebook) 
                    (inside ?notebook ?basket1)
                ) 
                (forall 
                    (?eraser - eraser) 
                    (inside ?eraser ?basket1)
                ) 
                (and 
                    (and 
                        (forall 
                            (?pen - pen) 
                            (inside ?pen ?bag1)
                        ) 
                        (forall 
                            (?pencil - pencil) 
                            (inside ?pencil ?bag1)
                        )
                    ) 
                    (not 
                        (open ?bag1)
                    ) 
                    (inside ?bag1 ?basket1)
                ) 
                (inside ?laptop1 ?basket1)
            ) 
            (not 
                (open ?basket1)
            )
        )
    )
)
