(define (problem assembling_gift_baskets_0)
    (:domain igibson)

    (:objects
     	; basket1 basket2 basket3 basket4 - basket
    	shelf1 - shelf
    	lotion1 lotion2 lotion3 lotion4 - lotion
    	bottom_cabinet1 - bottom_cabinet
    	soap1 soap2 soap3 soap4 - soap
    	shampoo1 shampoo2 shampoo3 shampoo4 - shampoo
    	conditioner1 conditioner2 conditioner3 conditioner4 - conditioner
    	envelope1 envelope2 envelope3 envelope4 - envelope
    	table1 - table
    )
    
    (:init 
        ; (and 
        ;     (ontop basket1 shelf1) 
        ;     (ontop basket2 shelf1) 
        ;     (ontop basket3 shelf1) 
        ;     (ontop basket4 shelf1)
        ; ) 
        (and 
            (inside lotion1 bottom_cabinet1) 
            (inside lotion2 bottom_cabinet1) 
            (inside lotion3 bottom_cabinet1) 
            (inside lotion4 bottom_cabinet1)
        ) 
        (and 
            (inside soap1 bottom_cabinet1) 
            (inside soap2 bottom_cabinet1) 
            (inside soap3 bottom_cabinet1) 
            (inside soap4 bottom_cabinet1)
        ) 
        (and 
            (inside shampoo1 bottom_cabinet1) 
            (inside shampoo2 bottom_cabinet1) 
            (inside shampoo3 bottom_cabinet1) 
            (inside shampoo4 bottom_cabinet1)
        ) 
        (and 
            (inside conditioner1 bottom_cabinet1) 
            (inside conditioner2 bottom_cabinet1) 
            (inside conditioner3 bottom_cabinet1) 
            (inside conditioner4 bottom_cabinet1)
        ) 
        (and 
            (ontop envelope1 table1) 
            (ontop envelope2 table1) 
            (ontop envelope3 table1) 
            (ontop envelope4 table1)
        )
        (inroom shelf1 living_room)
        (inroom bottom_cabinet1 living_room)
        (inroom table1 living_room)
    )
    
    (:goal 
        (and 
            (exists 
                (?table - table) 
                (forall 
                    (?basket - basket) 
                    (ontop ?basket ?table)
                )
            ) 
            (forpairs 
                (?basket - basket) 
                (?conditioner - conditioner) 
                (inside ?conditioner ?basket)
            ) 
            (forpairs 
                (?basket - basket) 
                (?shampoo - shampoo) 
                (inside ?shampoo ?basket)
            ) 
            (forpairs 
                (?basket - basket) 
                (?lotion - lotion) 
                (inside ?lotion ?basket)
            ) 
            (forpairs 
                (?basket - basket) 
                (?envelope - envelope) 
                (inside ?envelope ?basket)
            ) 
            (forpairs 
                (?basket - basket) 
                (?soap - soap) 
                (inside ?soap ?basket)
            )
        )
    )
)
