(define (problem cleaning_out_drawers_0)
    (:domain igibson)

    (:objects
        drawer1 drawer2 drawer3 drawer4 drawer5 drawer6 - drawer
        cabinet.n.01_1 - cabinet.n.01
        jewelry1 jewelry2 jewelry3 - jewelry
        jersey.n.03_1 jersey.n.03_2 jersey.n.03_3 jersey.n.03_4 jersey.n.03_5 - jersey.n.03
        jean.n.01_1 jean.n.01_2 jean.n.01_3 - jean.n.01
        dress1 dress2 dress3 dress4 - dress
        brassiere1 brassiere2 brassiere3 - brassiere
        underwear.n.01_1 underwear.n.01_2 underwear.n.01_3 - underwear.n.01
        sweater.n.01_1 sweater.n.01_2 sweater.n.01_3 sweater.n.01_4 - sweater.n.01
    )

    (:init
        (and
            (inside drawer1 cabinet.n.01_1)
            (inside drawer2 cabinet.n.01_1)
            (inside drawer3 cabinet.n.01_1)
            (inside drawer4 cabinet.n.01_1)
            (inside drawer5 cabinet.n.01_1)
            (inside drawer6 cabinet.n.01_1)
        )
        (and
            (inside jewelry1 drawer1)
            (inside jersey.n.03_4 drawer1)
            (inside jean.n.01_1 drawer1)
        )
        (and
            (inside dress1 drawer2)
            (inside brassiere3 drawer2)
            (inside jersey.n.03_1 drawer2)
            (inside brassiere1 drawer2)
        )
        (and
            (inside jersey.n.03_5 drawer3)
            (inside brassiere2 drawer3)
            (inside dress2 drawer3)
            (inside jean.n.01_2 drawer3)
        )
        (and
            (inside jewelry2 drawer4)
            (inside jersey.n.03_2 drawer4)
            (inside underwear.n.01_2 drawer4)
            (inside underwear.n.01_1 drawer4)
        )
        (and
            (inside dress3 drawer5)
            (inside sweater.n.01_1 drawer5)
            (inside underwear.n.01_3 drawer4)
            (inside jean.n.01_3 drawer5)
            (inside sweater.n.01_2 drawer5)
        )
        (and
            (inside jewelry3 drawer6)
            (inside dress4 drawer6)
            (inside sweater.n.01_3 drawer6)
            (inside sweater.n.01_4 drawer6)
            (inside jersey.n.03_3 drawer6)
        )
        (inside jewelry1 cabinet.n.01_1)
        (inside jersey.n.03_4 cabinet.n.01_1)
        (inside jean.n.01_1 cabinet.n.01_1)
        (inside dress1 cabinet.n.01_1)
        (inside brassiere3 cabinet.n.01_1)
        (inside jersey.n.03_1 cabinet.n.01_1)
        (inside brassiere1 cabinet.n.01_1)
        (inside jersey.n.03_5 cabinet.n.01_1)
        (inside brassiere2 cabinet.n.01_1)
        (inside dress2 cabinet.n.01_1)
        (inside jean.n.01_2 cabinet.n.01_1)
        (inside jewelry2 cabinet.n.01_1)
        (inside jersey.n.03_2 cabinet.n.01_1)
        (inside underwear.n.01_2 cabinet.n.01_1)
        (inside underwear.n.01_1 cabinet.n.01_1)
        (inside dress3 cabinet.n.01_1)
        (inside sweater.n.01_1 cabinet.n.01_1)
        (inside underwear.n.01_3 cabinet.n.01_1)
        (inside jean.n.01_3 cabinet.n.01_1)
        (inside sweater.n.01_2 cabinet.n.01_1)
        (inside jewelry3 cabinet.n.01_1)
        (inside dress4 cabinet.n.01_1)
        (inside sweater.n.01_3 cabinet.n.01_1)
        (inside sweater.n.01_4 cabinet.n.01_1)
        (inside jersey.n.03_3 cabinet.n.01_1)
        (inroom cabinet.n.01_1 bedroom)
    )

    (:goal
        (and
            (exists
                (?cabinet.n.01 - cabinet.n.01)
                (forall
                    (?drawer - drawer)
                    (inside ?drawer ?cabinet.n.01)
                )
            )
            (exists
                (?drawer - drawer)
                (forall
                    (?jewelry - jewelry)
                    (inside ?jewelry ?drawer)
                )
            )
            (exists
                (?drawer - drawer)
                (and
                    (forall
                        (?brassiere - brassiere)
                        (inside ?brassiere ?drawer)
                    )
                    (forall
                        (?underwear.n.01 - underwear.n.01)
                        (inside ?underwear.n.01 ?drawer)
                    )
                )
            )
            (exists
                (?drawer - drawer)
                (forall
                    (?dress - dress)
                    (inside ?dress ?drawer)
                )
            )
            (exists
                (?drawer - drawer)
                (forall
                    (?sweater.n.01 - sweater.n.01)
                    (inside ?sweater.n.01 ?drawer)
                )
            )
            (exists
                (?drawer - drawer)
                (and
                    (forall
                        (?jersey.n.03 - jersey.n.03)
                        (inside ?jersey.n.03 ?drawer)
                    )
                    (forall
                        (?jean.n.01 - jean.n.01)
                        (inside ?jean.n.01 ?drawer)
                    )
                )
            )
        )
)
