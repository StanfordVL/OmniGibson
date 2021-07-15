(define (problem sorting_books_0)
    (:domain igibson)

    (:objects
     	hardback.n.01_1 hardback.n.01_2 - hardback.n.01
    	table.n.02_1 - table.n.02
    	floor.n.01_1 - floor.n.01
    	shelf.n.01_1 - shelf.n.01
    	book.n.02_1 book.n.02_2 - book.n.02
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (ontop hardback.n.01_1 table.n.02_1) 
        (onfloor hardback.n.01_2 floor.n.01_1) 
        (onfloor book.n.02_1 floor.n.01_1) 
        (ontop book.n.02_2 table.n.02_1) 
        (inroom table.n.02_1 living_room) 
        (inroom shelf.n.01_1 living_room) 
        (inroom floor.n.01_1 living_room) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forall 
                (?hardback.n.01 - hardback.n.01) 
                (ontop ?hardback.n.01 ?shelf.n.01_1)
            ) 
            (forall 
                (?book.n.02 - book.n.02) 
                (ontop ?book.n.02 ?shelf.n.01_1)
            )
        )
    )
)