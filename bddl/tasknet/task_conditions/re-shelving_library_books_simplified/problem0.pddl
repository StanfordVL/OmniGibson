(define 
(problem sampling_test)
(:domain igibson)
(:objects
    floor.n.01_1 - floor.n.01
    agent.n.01_1 - agent.n.01
    table.n.02_1 - table.n.02
    shelf.n.01_1 - shelf.n.01
    book.n.02_1 book.n.02_2 book.n.02_3 - book.n.02
    plaything.n.01_1 plaything.n.01_2 plaything.n.01_3 - plaything.n.01
    laptop.n.01_1 - laptop.n.01
    cup.n.01_1 - cup.n.01
    headset.n.01_1 - headset.n.01
    mouse.n.04_1 - mouse.n.04
    pen.n.01_1 - pen.n.01
    pencil.n.01_1 - pencil.n.01
    chair.n.01_1 - chair.n.01
)
(:init 
    (inroom floor.n.01_1 kitchen)
    (inroom table.n.02_1 living_room)
    (inroom shelf.n.01_1 living_room)
    (inroom chair.n.01_1 living_room)
    (onfloor agent.n.01_1 floor.n.01_1)
    (ontop laptop.n.01_1 table.n.02_1)
    (ontop book.n.02_1 table.n.02_1)
    (ontop book.n.02_2 table.n.02_1)
    (ontop headset.n.01_1 table.n.02_1)
    (ontop cup.n.01_1 table.n.02_1)
    (ontop mouse.n.04_1 table.n.02_1)
    (ontop pen.n.01_1 table.n.02_1)
    (ontop pencil.n.01_1 table.n.02_1)
    (ontop book.n.02_3 chair.n.01_1)
    (inside plaything.n.01_1 shelf.n.01_1)
    (inside plaything.n.01_2 shelf.n.01_1)
    (inside plaything.n.01_3 shelf.n.01_1)
)
(:goal 
    (and 
        (inside book.n.02_1 shelf.n.01_1)
        (inside book.n.02_2 shelf.n.01_1)
        (inside book.n.02_3 shelf.n.01_1)
    )
)
)
