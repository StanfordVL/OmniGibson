(define (problem assembling_gift_baskets_0)
    (:domain igibson)

    (:objects
     	basket1 basket2 basket3 basket4 - basket
    	shelf1 - shelf
    	lotion1 lotion2 lotion3 lotion4 - lotion
    	top_cabinet1 - top_cabinet
    	soap1 soap2 soap3 soap4 - soap
    	shampoo1 shampoo2 shampoo3 shampoo4 - shampoo
    	conditioner1 conditioner2 conditioner3 conditioner4 - conditioner
    	photograph1 photograph2 photograph3 photograph4 - photograph
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
            (inside lotion1 top_cabinet1) 
            (inside lotion2 top_cabinet1) 
            (inside lotion3 top_cabinet1) 
            (inside lotion4 top_cabinet1)
        ) 
        (and 
            (inside soap1 top_cabinet1) 
            (inside soap2 top_cabinet1) 
            (inside soap3 top_cabinet1) 
            (inside soap4 top_cabinet1)
        ) 
        (and 
            (inside shampoo1 top_cabinet1) 
            (inside shampoo2 top_cabinet1) 
            (inside shampoo3 top_cabinet1) 
            (inside shampoo4 top_cabinet1)
        ) 
        (and 
            (inside conditioner1 top_cabinet1) 
            (inside conditioner2 top_cabinet1) 
            (inside conditioner3 top_cabinet1) 
            (inside conditioner4 top_cabinet1)
        ) 
        (and 
            (ontop photograph1 table1) 
            (ontop photograph2 table1) 
            (ontop photograph3 table1) 
            (ontop photograph4 table1)
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
                (?photograph - photograph) 
                (inside ?photograph ?basket)
            ) 
            (forpairs 
                (?basket - basket) 
                (?soap - soap) 
                (inside ?soap ?basket)
            )
        )
    )
)
