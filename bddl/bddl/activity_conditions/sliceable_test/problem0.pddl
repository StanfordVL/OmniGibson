(define (problem packlunch)
    (:domain igibson)
    (:objects
        table.n.02_1 - table.n.02
        strawberry.n.01_1 - strawberry.n.01
        carving_knife.n.01_1 - carving_knife.n.01
        plate.n.04_1 - plate.n.04
    )
    (:init
        (inRoom table.n.02_1 living_room)
        (onTop strawberry.n.01_1 table.n.02_1)
        (onTop carving_knife.n.01_1 table.n.02_1)
        (onTop plate.n.04_1 table.n.02_1)
    )
    (:goal
        (and 
            (sliced strawberry.n.01_1)
            (onTop strawberry.n.01_1 plate.n.04_1)
        )
    )
)
