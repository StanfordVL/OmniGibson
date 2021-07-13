(define (problem cleaning_out_drawers_0)
    (:domain igibson)

    (:objects
        cabinet.n.01_1 - cabinet.n.01
        jean.n.01_1 - jean.n.01
        hat.n.01_1 - hat.n.01
        scarf.n.01_1 - scarf.n.01
        bed.n.01_1 - bed.n.01
    )

    (:init
        (inside jean.n.01_1 cabinet.n.01_1)
        (inside hat.n.01_1 cabinet.n.01_1)
        (inside scarf.n.01_1 cabinet.n.01_1)
        (inroom cabinet.n.01_1 bedroom)
        (inroom bed.n.01_1 bedroom)
    )

    (:goal
        (and
            (forall
                (?jean.n.01 - jean.n.01)
                (ontop ?jean.n.01 ?bed.n.01_1)
            )
            (forall
                (?hat.n.01 - hat.n.01)
                (ontop ?hat.n.01 ?bed.n.01_1)
            )
            (forall
                (?scarf.n.01 - scarf.n.01)
                (ontop ?scarf.n.01 ?bed.n.01_1)
            )
        )
    )
)
