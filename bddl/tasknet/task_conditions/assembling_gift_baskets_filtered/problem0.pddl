(define 
    (problem assembling_gift_baskets_0)
    (:domain igibson)

    (:objects
        cabinet.n.01_1 - cabinet.n.01
    	table.n.02_1 - table.n.02
        ; shelf.n.01_1 - shelf.n.01
        basket.n.01_1 basket.n.01_2 - basket.n.01
        lotion.n.01_1 lotion.n.01_2 - lotion.n.01
        soap.n.01_1 soap.n.01_2 - soap.n.01
        shampoo.n.01_1 shampoo.n.01_2 - shampoo.n.01
        envelope.n.01_1 envelope.n.01_2 - envelope.n.01
    )
    
    (:init 
        (ontop basket.n.01_1 table.n.02_1)
        (ontop basket.n.01_2 table.n.02_1)
        (ontop lotion.n.01_1 table.n.02_1)
        (ontop lotion.n.01_2 table.n.02_1)
        (ontop shampoo.n.01_1 table.n.02_1)
        (ontop shampoo.n.01_2 table.n.02_1)
        (inside soap.n.01_1 cabinet.n.01_1)
        (inside soap.n.01_2 cabinet.n.01_1)
        (inside envelope.n.01_1 cabinet.n.01_1)
        (inside envelope.n.01_2 cabinet.n.01_1)
        ; (inroom shelf.n.01_1 living_room)
        (inroom cabinet.n.01_1 living_room)
        (inroom table.n.02_1 living_room)
    )
    
    (:goal 
        (and 
            (exists 
                (?table.n.02 - table.n.02) 
                (forall 
                    (?basket.n.01 - basket.n.01) 
                    (ontop ?basket.n.01 ?table.n.02)
                )
            ) 
            (forpairs 
                (?basket.n.01 - basket.n.01) 
                (?shampoo.n.01 - shampoo.n.01) 
                (inside ?shampoo.n.01 ?basket.n.01)
            ) 
            (forpairs 
                (?basket.n.01 - basket.n.01) 
                (?lotion.n.01 - lotion.n.01) 
                (inside ?lotion.n.01 ?basket.n.01)
            ) 
            (forpairs 
                (?basket.n.01 - basket.n.01) 
                (?envelope.n.01 - envelope.n.01) 
                (inside ?envelope.n.01 ?basket.n.01)
            ) 
            (forpairs 
                (?basket.n.01 - basket.n.01) 
                (?soap.n.01 - soap.n.01) 
                (inside ?soap.n.01 ?basket.n.01)
            )
        )
    )
)
