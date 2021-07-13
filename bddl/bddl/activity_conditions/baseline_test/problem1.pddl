(define (problem packlunch)
    (:domain igibson)
    (:objects
        chip.n.04_1 chip.n.04_2 chip.n.04_3 - chip.n.04
        table.n.02_1 table.n.02_2 - table.n.02
    )
    (:init
        (inRoom table.n.02_1 living_room)
        (inRoom table.n.02_2 living_room)
        (onTop chip.n.04_1 table.n.02_1)
        (onTop chip.n.04_2 table.n.02_1)
        (onTop chip.n.04_3 table.n.02_1)
    )
    (:goal
        (and 
        (onTop chip.n.04_1 table.n.02_2)
        (onTop chip.n.04_2 table.n.02_2)
        (onTop chip.n.04_3 table.n.02_2)
        )
    )
)
