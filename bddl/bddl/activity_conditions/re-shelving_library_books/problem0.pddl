(define (problem re-shelving_library_books_0)
    (:domain igibson)

    (:objects
     	book.n.02_1 book.n.02_2 book.n.02_3 book.n.02_4 book.n.02_5 book.n.02_6 book.n.02_7 book.n.02_8 - book.n.02
    	table.n.02_1 - table.n.02
    	shelf.n.01_1 - shelf.n.01
    	floor.n.01_1 - floor.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (ontop book.n.02_1 table.n.02_1) 
        (ontop book.n.02_2 table.n.02_1) 
        (ontop book.n.02_3 table.n.02_1) 
        (ontop book.n.02_4 table.n.02_1) 
        (ontop book.n.02_5 table.n.02_1) 
        (ontop book.n.02_6 table.n.02_1) 
        (ontop book.n.02_7 table.n.02_1) 
        (ontop book.n.02_8 table.n.02_1) 
        (inroom table.n.02_1 living_room) 
        (inroom shelf.n.01_1 living_room) 
        (inroom floor.n.01_1 living_room) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (ontop ?book.n.02_1 ?shelf.n.01_1) 
            (ontop ?book.n.02_2 ?shelf.n.01_1) 
            (ontop ?book.n.02_3 ?shelf.n.01_1) 
            (ontop ?book.n.02_4 ?shelf.n.01_1) 
            (ontop ?book.n.02_5 ?table.n.02_1) 
            (ontop ?book.n.02_6 ?table.n.02_1) 
            (ontop ?book.n.02_7 ?table.n.02_1) 
            (ontop ?book.n.02_8 ?table.n.02_1)
        )
    )
)