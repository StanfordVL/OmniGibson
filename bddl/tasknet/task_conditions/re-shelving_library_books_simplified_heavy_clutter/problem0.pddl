(define 
(problem sampling_test)
(:domain igibson)
(:objects
    floor.n.01_1 - floor.n.01
    agent.n.01_1 - agent.n.01
    table.n.02_1 - table.n.02
    shelf.n.01_1 - shelf.n.01
    book.n.02_1 book.n.02_2 book.n.02_3 - book.n.02
    cup.n.01_1 cup.n.01_2 cup.n.01_3 cup.n.01_4 - cup.n.01
    headset.n.01_1 - headset.n.01
    mouse.n.04_1 - mouse.n.04
    keyboard.n.01_1 - keyboard.n.01
    pen.n.01_1 - pen.n.01
    pencil.n.01_1 - pencil.n.01
    chair.n.01_1 - chair.n.01
    hand_towel.n.01_1 - hand_towel.n.01
    cracker.n.01_1 - cracker.n.01
    comb.n.01_1 - comb.n.01
    computer_game.n.01_1 - computer_game.n.01
    screwdriver.n.01_1 - screwdriver.n.01
    orange.n.01_1 orange.n.01_2 orange.n.01_3 - orange.n.01
    scrub_brush.n.01_1 - scrub_brush.n.01
    calculator.n.02_1 - calculator.n.02
    apple.n.01_1 - apple.n.01
    perfume.n.02_1 - perfume.n.02
)
(:init 
    (inroom floor.n.01_1 kitchen)
    (inroom table.n.02_1 living_room)
    (inroom shelf.n.01_1 living_room)
    (inroom chair.n.01_1 living_room)
    (onfloor agent.n.01_1 floor.n.01_1)
    (ontop book.n.02_1 table.n.02_1)
    (ontop book.n.02_2 table.n.02_1)
    (ontop headset.n.01_1 table.n.02_1)
    (ontop cup.n.01_1 table.n.02_1)
    (ontop mouse.n.04_1 table.n.02_1)
    (ontop pen.n.01_1 table.n.02_1)
    (ontop pencil.n.01_1 table.n.02_1)
    (ontop keyboard.n.01_1 table.n.02_1)
    (ontop apple.n.01_1 table.n.02_1)
    (ontop book.n.02_3 chair.n.01_1)
    (inside cup.n.01_2 shelf.n.01_1)
    (inside cup.n.01_3 shelf.n.01_1)
    (inside cup.n.01_4 shelf.n.01_1)
    (inside hand_towel.n.01_1 shelf.n.01_1)
    (inside cracker.n.01_1 shelf.n.01_1)
    (inside comb.n.01_1 shelf.n.01_1)
    (inside computer_game.n.01_1 shelf.n.01_1)
    (inside screwdriver.n.01_1 shelf.n.01_1)
    (inside orange.n.01_1 shelf.n.01_1)
    (inside orange.n.01_2 shelf.n.01_1)
    (inside orange.n.01_3 shelf.n.01_1)
    (inside scrub_brush.n.01_1 shelf.n.01_1)
    (inside calculator.n.02_1 shelf.n.01_1)
    (inside perfume.n.02_1 shelf.n.01_1)
)
(:goal 
    (and 
        (inside book.n.02_1 shelf.n.01_1)
        (inside book.n.02_2 shelf.n.01_1)
        (inside book.n.02_3 shelf.n.01_1)
    )
)
)
