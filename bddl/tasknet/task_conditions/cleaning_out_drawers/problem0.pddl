(define (problem cleaning_out_drawers_0)
    (:domain igibson)

    (:objects
        drawer1 drawer2 drawer3 drawer4 drawer5 drawer6 - drawer
        cabinet1 - cabinet
        jewelry1 jewelry2 jewelry3 - jewelry
        shirt1 shirt2 shirt3 shirt4 shirt5 - shirt
        jean1 jean2 jean3 - jean
        dress1 dress2 dress3 dress4 - dress
        brassiere1 brassiere2 brassiere3 - brassiere
        underwear1 underwear2 underwear3 - underwear
        sweater1 sweater2 sweater3 sweater4 - sweater
    )
    
    (:init 
        (and 
            (inside drawer1 cabinet1) 
            (inside drawer2 cabinet1) 
            (inside drawer3 cabinet1) 
            (inside drawer4 cabinet1) 
            (inside drawer5 cabinet1) 
            (inside drawer6 cabinet1)
        ) 
        (and 
            (inside jewelry1 drawer1) 
            (inside shirt4 drawer1) 
            (inside jean1 drawer1)
        ) 
        (and 
            (inside dress1 drawer2) 
            (inside brassiere3 drawer2) 
            (inside shirt1 drawer2) 
            (inside brassiere1 drawer2)
        ) 
        (and 
            (inside shirt5 drawer3) 
            (inside brassiere2 drawer3) 
            (inside dress2 drawer3) 
            (inside jean2 drawer3)
        ) 
        (and 
            (inside jewelry2 drawer4) 
            (inside shirt2 drawer4) 
            (inside underwear2 drawer4) 
            (inside underwear1 drawer4)
        ) 
        (and 
            (inside dress3 drawer5) 
            (inside sweater1 drawer5) 
            (inside underwear3 drawer4) 
            (inside jean3 drawer5) 
            (inside sweater2 drawer5)
        ) 
        (and 
            (inside jewelry3 drawer6) 
            (inside dress4 drawer6) 
            (inside sweater3 drawer6) 
            (inside sweater4 drawer6) 
            (inside shirt3 drawer6)
        ) 
        (inside jewelry1 cabinet1) 
        (inside shirt4 cabinet1) 
        (inside jean1 cabinet1) 
        (inside dress1 cabinet1) 
        (inside brassiere3 cabinet1) 
        (inside shirt1 cabinet1) 
        (inside brassiere1 cabinet1) 
        (inside shirt5 cabinet1) 
        (inside brassiere2 cabinet1) 
        (inside dress2 cabinet1) 
        (inside jean2 cabinet1) 
        (inside jewelry2 cabinet1) 
        (inside shirt2 cabinet1) 
        (inside underwear2 cabinet1) 
        (inside underwear1 cabinet1) 
        (inside dress3 cabinet1) 
        (inside sweater1 cabinet1) 
        (inside underwear3 cabinet1) 
        (inside jean3 cabinet1) 
        (inside sweater2 cabinet1) 
        (inside jewelry3 cabinet1) 
        (inside dress4 cabinet1) 
        (inside sweater3 cabinet1) 
        (inside sweater4 cabinet1) 
        (inside shirt3 cabinet1) 
        (inroom cabinet1 bedroom)
    )
    
    (:goal 
        (and 
            (exists 
                (?cabinet - cabinet) 
                (forall 
                    (?drawer - drawer) 
                    (inside ?drawer ?cabinet)
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
                        (?underwear - underwear) 
                        (inside ?underwear ?drawer)
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
                    (?sweater - sweater) 
                    (inside ?sweater ?drawer)
                )
            ) 
            (exists 
                (?drawer - drawer) 
                (and 
                    (forall 
                        (?shirt - shirt) 
                        (inside ?shirt ?drawer)
                    ) 
                    (forall 
                        (?jean - jean) 
                        (inside ?jean ?drawer)
                    )
                )
            )
        )
)