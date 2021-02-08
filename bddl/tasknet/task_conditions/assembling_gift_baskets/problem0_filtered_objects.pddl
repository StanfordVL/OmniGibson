(define (problem assembling_gift_baskets_0)
    (:domain igibson)

    (:objects
     	hamper1 hamper2 hamper3 hamper4 - hamper
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
            (ontop hamper1 shelf1) 
            (ontop hamper2 shelf1) 
            (ontop hamper3 shelf1) 
            (ontop hamper4 shelf1)
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
                    (?hamper - hamper) 
                    (ontop ?hamper ?table)
                )
            ) 
            (forpairs 
                (?hamper - hamper) 
                (?conditioner - conditioner) 
                (inside ?conditioner ?hamper)
            ) 
            (forpairs 
                (?hamper - hamper) 
                (?shampoo - shampoo) 
                (inside ?shampoo ?hamper)
            ) 
            (forpairs 
                (?hamper - hamper) 
                (?lotion - lotion) 
                (inside ?lotion ?hamper)
            ) 
            (forpairs 
                (?hamper - hamper) 
                (?photograph - photograph) 
                (inside ?photograph ?hamper)
            ) 
            (forpairs 
                (?hamper - hamper) 
                (?soap - soap) 
                (inside ?soap ?hamper)
            )
        )
    )
)
