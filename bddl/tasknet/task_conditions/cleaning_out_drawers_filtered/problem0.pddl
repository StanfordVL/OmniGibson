(define (problem cleaning_out_drawers_0)
    (:domain igibson)

    (:objects
        cabinet.n.01_1 - cabinet.n.01
        jean.n.01_1 jean.n.01_2 jean.n.01_3 - jean.n.01
        underwear.n.01_1 underwear.n.01_2 underwear.n.01_3 - underwear.n.01
        bed.n.01_1 - bed.n.01
    )

    (:init
        (inside jean.n.01_1 cabinet.n.01_1)
        (inside jean.n.01_2 cabinet.n.01_1)
        (inside jean.n.01_3 cabinet.n.01_1)
        (inside underwear.n.01_2 cabinet.n.01_1)
        (inside underwear.n.01_1 cabinet.n.01_1)
        (inside underwear.n.01_3 cabinet.n.01_1)
        (inroom cabinet.n.01_1 bedroom)
        (inroom bed.n.01_1 bedroom)
    )

    (:goal
        (and
            (exists
                (?cabinet.n.01 - cabinet.n.01)
                (forall
                    (?underwear.n.01 - underwear.n.01)
                    (ontop ?underwear.n.01 ?bed.n.01_1)
                )
            )
        )
        (and
            (exists
                (?cabinet.n.01 - cabinet.n.01)
                (forall
                    (?jean.n.01 - jean.n.01)
                    (ontop ?jean.n.01 ?bed.n.01_1)
                )
            )
        )
    )
)
