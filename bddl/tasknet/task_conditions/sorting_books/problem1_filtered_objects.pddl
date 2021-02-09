(define (problem sorting_books_1)
    (:domain igibson)

    (:objects
     	notebook1 notebook2 notebook3 notebook4 notebook5 notebook6 notebook7 - notebook
    	table1 - table
    	shelf1 - shelf
    	chair1 - chair
    	hardback1 hardback2 hardback3 hardback4 hardback5 - hardback
    )
    
    (:init 
        (ontop notebook1 table1) 
        (ontop notebook2 shelf1) 
        (ontop notebook3 chair1) 
        (ontop notebook4 table1) 
        (ontop notebook5 chair1) 
        (ontop hardback1 table1) 
        (ontop hardback2 shelf1) 
        (ontop hardback3 shelf1) 
        (ontop hardback4 shelf1) 
        (ontop notebook6 chair1) 
        (ontop notebook7 table1) 
        (ontop hardback5 shelf1) 
        (inroom table1 living room) 
        (inroom shelf1 living room) 
        (inroom chair1 living room)
    )
    
    (:goal 
        (and 
            (forall 
                (?notebook - notebook) 
                (or 
                    (nextto ?notebook ?notebook1) 
                    (nextto ?notebook ?notebook2) 
                    (nextto ?notebook ?notebook3) 
                    (nextto ?notebook ?notebook4) 
                    (nextto ?notebook ?notebook5)
                    (nextto ?notebook ?notebook6)
                    (nextto ?notebook ?notebook7)
                )
            ) 
            (forall 
                (?hardback - hardback) 
                (or 
                    (nextto ?hardback ?hardback1) 
                    (nextto ?hardback ?hardback2) 
                    (nextto ?hardback ?hardback3) 
                    (nextto ?hardback ?hardback4)
                    (nextto ?hardback ?hardback5)
                )
            ) 
            (scrubbed ?table1)
        )
    )
)