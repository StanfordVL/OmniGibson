(define (problem cleaning_out_drawers_1)
    (:domain igibson)

    (:objects
    	pajama.n.02_1 - pajama.n.02
        sock.n.01_1 sock.n.01_2 sock.n.01_3 sock.n.01_4 - sock.n.01
        underwear.n.01_1 underwear.n.01_2 - underwear.n.01
    	cabinet.n.01_1 - cabinet.n.01
    	bed.n.01_1 - bed.n.01
    )

    (:init
        (inside pajama.n.02_1 cabinet.n.01_1)
        (inside sock.n.01_1 cabinet.n.01_1)
        (inside sock.n.01_2 cabinet.n.01_1)
        (inside sock.n.01_3 cabinet.n.01_1)
        (inside sock.n.01_4 cabinet.n.01_1)
        (inside underwear.n.01_1 cabinet.n.01_1)
        (inside underwear.n.01_2 cabinet.n.01_1)
        (inroom cabinet.n.01_1 bedroom)
        (inroom bed.n.01_1 bedroom)
    )

    (:goal
        (and
            (forall
                (?pajama.n.02 - pajama.n.02)
                (ontop ?pajama.n.02 ?bed.n.01_1)
            )
            (forall
                (?sock.n.01 - sock.n.01)
                (ontop ?sock.n.01 ?bed.n.01_1)
            )
            (forall
                (?underwear.n.01 - underwear.n.01)
                (ontop ?underwear.n.01 ?bed.n.01_1)
            )
        )
    )
)
