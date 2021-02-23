(define (problem putting_away_toys_0)
    (:domain igibson)

    (:objects
        plaything.n.01_1 plaything.n.01_2 plaything.n.01_3 - plaything.n.01
        sofa.n.01_1 - sofa.n.01
        table.n.02_1 - table.n.02
        shelf.n.01_1 shelf.n.01_2 - shelf.n.01
    )
    
    (:init
        (ontop plaything.n.01_1 sofa.n.01_1)
        (under plaything.n.01_2 table.n.02_1)
        (ontop plaything.n.01_3 table.n.02_1)
        (inroom sofa.n.01_1 living_room)
        (inroom table.n.02_1 living_room)
        (inroom shelf.n.01_1 living_room)
        (inroom shelf.n.01_2 living_room)
    )
    
    (:goal
        (and
            (exists
                (?shelf.n.01 - shelf.n.01)
                (forall
                    (?plaything.n.01 - plaything.n.01)
                    (inside ?plaything.n.01 ?shelf.n.01)
                )
            )
        )
    )
)