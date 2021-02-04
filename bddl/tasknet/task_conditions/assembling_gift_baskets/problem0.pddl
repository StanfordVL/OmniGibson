(define (problem assembling_gift_baskets_0
    (:domain igibson)

    (:objects
     	basket1 basket2 basket3 basket4 - basket
    	shelf1 - shelf
    	lotion1 lotion2 lotion3 lotion4 - lotion
    	cabinet1 - cabinet
    	soap1 soap2 soap3 soap4 - soap
    	shampoo1 shampoo2 shampoo3 shampoo4 - shampoo
    	conditioner1 conditioner2 conditioner3 conditioner4 - conditioner
    	card1 card2 card3 card4 - card
    	table1 - table
    )
    
    (:init 
        (and 
            (ontop basket1 shelf1) 
            (ontop basket2 shelf1) 
            (ontop basket3 shelf1) 
            (ontop basket4 shelf1)
        ) 
        (and 
            (inside lotion1 cabinet1) 
            (inside lotion2 cabinet1) 
            (inside lotion3 cabinet1) 
            (inside lotion4 cabinet1)
        ) 
        (and 
            (inside soap1 cabinet1) 
            (inside soap2 cabinet1) 
            (inside soap3 cabinet1) 
            (inside soap4 cabinet1)
        ) 
        (and 
            (inside shampoo1 cabinet1) 
            (inside shampoo2 cabinet1) 
            (inside shampoo3 cabinet1) 
            (inside shampoo4 cabinet1)
        ) 
        (and 
            (inside conditioner1 cabinet1) 
            (inside conditioner2 cabinet1) 
            (inside conditioner3 cabinet1) 
            (inside conditioner4 cabinet1)
        ) 
        (and 
            (ontop card1 table1) 
            (ontop card2 table1) 
            (ontop card3 table1) 
            (ontop card4 table1)
        )
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
                (?card - card) 
                (inside ?card ?basket)
            ) 
            (forpairs 
                (?basket - basket) 
                (?soap - soap) 
                (inside ?soap ?basket)
            )
        )
    )
)