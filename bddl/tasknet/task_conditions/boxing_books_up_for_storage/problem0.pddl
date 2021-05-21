(define (problem boxing_books_up_for_storage_0)
    (:domain igibson)

    (:objects
     	book.n.02_1 book.n.02_2 book.n.02_3 book.n.02_4 book.n.02_5 book.n.02_6 book.n.02_7 book.n.02_8 - book.n.02
    	floor.n.01_1 - floor.n.01
    	shelf.n.01_1 - shelf.n.01
    	box.n.01_1 box.n.01_2 - box.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor book.n.02_1 floor.n.01_1) 
        (onfloor book.n.02_2 floor.n.01_1) 
        (onfloor book.n.02_3 floor.n.01_1) 
        (onfloor book.n.02_4 floor.n.01_1) 
        (onfloor book.n.02_5 floor.n.01_1) 
        (ontop book.n.02_6 shelf.n.01_1) 
        (ontop book.n.02_7 shelf.n.01_1) 
        (ontop book.n.02_8 box.n.01_1) 
        (onfloor box.n.01_1 floor.n.01_1) 
        (onfloor box.n.01_2 floor.n.01_1) 
        (inroom floor.n.01_1 living_room) 
        (inroom shelf.n.01_1 living_room) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (and 
                (exists 
                    (?book.n.02 - book.n.02) 
                    (fornpairs 
                        (2) 
                        (?book.n.02 - book.n.02) 
                        (?box.n.01 - box.n.01) 
                        (inside ?book.n.02 ?box.n.01_1)
                    )
                ) 
                (exists 
                    (?box.n.01 - box.n.01) 
                    (fornpairs 
                        (2) 
                        (?book.n.02 - book.n.02) 
                        (?box.n.01 - box.n.01) 
                        (inside ?book.n.02 ?box.n.01_2)
                    )
                )
            )
        )
    )
)