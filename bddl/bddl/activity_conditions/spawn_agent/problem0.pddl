(define (problem re-shelving_library_books_0)
    (:domain igibson)

    (:objects
        book.n.02_1 book.n.02_2 - book.n.02
        table.n.02_1 - table.n.02
        shelf.n.01_1 - shelf.n.01
        floor.n.01_1 - floor.n.01
        agent.n.01_1 - agent.n.01
    )
    

    (:init 
        (inRoom floor.n.01_1 living_room)
        (onFloor agent.n.01_1 floor.n.01_1)
        (ontop book.n.02_1 table.n.02_1)
        (ontop book.n.02_2 table.n.02_1)
        (inroom table.n.02_1 living_room)
        (inroom shelf.n.01_1 living_room)
    )

    (:goal 
        (and 
            (forall
                (?book.n.02 - book.n.02)
                (or
                    (inside ?book.n.02 ?shelf.n.01_1)
                    (ontop ?book.n.02 ?shelf.n.01_1)
                )
            )
            (forall
                (?book.n.02 - book.n.02)
                (not
                    (ontop ?book.n.02 ?table.n.02_1)
                )
            )
        )
    )
)
