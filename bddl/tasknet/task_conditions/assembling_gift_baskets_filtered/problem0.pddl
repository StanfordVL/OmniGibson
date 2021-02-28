(define (problem assembling_gift_baskets_0)
    (:domain igibson)

    (:objects
     	; basket.n.01_1 basket.n.01_2 basket.n.01_3 basket.n.01_4 - basket.n.01
        basket.n.01_1 - basket.n.01
    	shelf.n.01_1 - shelf.n.01
    	; lotion.n.01_1 lotion.n.01_2 lotion.n.01_3 lotion.n.01_4 - lotion.n.01
        lotion.n.01_1 - lotion.n.01
    	cabinet.n.01_1 - cabinet.n.01
    	; soap.n.01_1 soap.n.01_2 soap.n.01_3 soap.n.01_4 - soap.n.01
        soap.n.01_1 - soap.n.01
    	; shampoo.n.01_1 shampoo.n.01_2 shampoo.n.01_3 shampoo.n.01_4 - shampoo.n.01
        shampoo.n.01_1 - shampoo.n.01 
    	; conditioner.n.03_1 conditioner.n.03_2 conditioner.n.03_3 conditioner.n.03_4 - conditioner.n.03
        conditioner.n.03_1 - conditioner.n.03
    	; photograph1 photograph2 photograph3 photograph4 - photograph
    	; envelope.n.01_1 envelope.n.01_2 envelope.n.01_3 envelope.n.01_4 - envelope.n.01
        envelope.n.01_1 - envelope.n.01
    	table.n.02_1 - table.n.02
    )
    
    (:init 
        (ontop basket.n.01_1 table.n.02_1)
        (inside lotion.n.01_1 cabinet.n.01_1) 
        ; (inside lotion.n.01_2 cabinet.n.01_1) 
        ; (inside lotion.n.01_3 cabinet.n.01_1) 
        ; (inside lotion.n.01_4 cabinet.n.01_1)
        (inside soap.n.01_1 cabinet.n.01_1) 
        ; (inside soap.n.01_2 cabinet.n.01_1) 
        ; (inside soap.n.01_3 cabinet.n.01_1) 
        ; (inside soap.n.01_4 cabinet.n.01_1)
        (inside shampoo.n.01_1 cabinet.n.01_1) 
        ; (inside shampoo.n.01_2 cabinet.n.01_1) 
        ; (inside shampoo.n.01_3 cabinet.n.01_1) 
        ; (inside shampoo.n.01_4 cabinet.n.01_1)
        (inside conditioner.n.03_1 cabinet.n.01_1) 
        ; (inside conditioner.n.03_2 cabinet.n.01_1) 
        ; (inside conditioner.n.03_3 cabinet.n.01_1) 
        ; (inside conditioner.n.03_4 cabinet.n.01_1)
        (ontop envelope.n.01_1 table.n.02_1) 
        ; (ontop envelope.n.01_2 table.n.02_1) 
        ; (ontop envelope.n.01_3 table.n.02_1) 
        ; (ontop envelope.n.01_4 table.n.02_1)
        (inroom shelf.n.01_1 living_room)
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
                (?conditioner.n.03 - conditioner.n.03) 
                (inside ?conditioner.n.03 ?basket.n.01)
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
