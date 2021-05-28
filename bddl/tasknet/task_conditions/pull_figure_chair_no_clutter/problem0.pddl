(define 
(problem sampling_test)
(:domain igibson)
(:objects
	floor.n.01_1 - floor.n.01
    agent.n.01_1 - agent.n.01
    table.n.02_1 - table.n.02
    shelf.n.01_1 - shelf.n.01
    book.n.02_1 book.n.02_2 book.n.02_3 - book.n.02
    chair.n.01_1 - chair.n.01
)
(:init 
    (inroom floor.n.01_1 kitchen)
    (inroom table.n.02_1 living_room)
    (inroom shelf.n.01_1 living_room)
    (inroom chair.n.01_1 living_room)
    (onfloor agent.n.01_1 floor.n.01_1)
    (ontop book.n.02_1 table.n.02_1)
    (ontop book.n.02_2 table.n.02_1)
    (ontop book.n.02_3 chair.n.01_1)
)
(:goal 
    (and 
        (inside book.n.02_1 shelf.n.01_1)
        (inside book.n.02_2 shelf.n.01_1)
        (inside book.n.02_3 shelf.n.01_1)
    )
)
)