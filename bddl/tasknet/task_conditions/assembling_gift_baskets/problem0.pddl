(define (problem assembling_gift_baskets_0)
    (:domain igibson)

    (:objects
     	wrapping.n.01_1 wrapping.n.01_2 wrapping.n.01_3 wrapping.n.01_4 - wrapping.n.01
    	floor.n.01_1 - floor.n.01
    	candy.n.01_1 candy.n.01_2 candy.n.01_3 candy.n.01_4 - candy.n.01
    	basket.n.01_1 basket.n.01_2 basket.n.01_3 basket.n.01_4 - basket.n.01
    	cookie.n.01_1 cookie.n.01_2 cookie.n.01_3 cookie.n.01_4 - cookie.n.01
    	book.n.02_1 book.n.02_2 book.n.02_3 book.n.02_4 - book.n.02
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor wrapping.n.01_2 floor.n.01_1) 
        (onfloor wrapping.n.01_3 floor.n.01_1) 
        (onfloor wrapping.n.01_4 floor.n.01_1) 
        (onfloor candy.n.01_1 floor.n.01_1) 
        (onfloor candy.n.01_2 floor.n.01_1) 
        (onfloor candy.n.01_3 floor.n.01_1) 
        (onfloor candy.n.01_4 floor.n.01_1) 
        (onfloor basket.n.01_1 floor.n.01_1) 
        (onfloor basket.n.01_2 floor.n.01_1) 
        (onfloor basket.n.01_3 floor.n.01_1) 
        (onfloor basket.n.01_4 floor.n.01_1) 
        (onfloor cookie.n.01_1 floor.n.01_1) 
        (onfloor cookie.n.01_2 floor.n.01_1) 
        (onfloor cookie.n.01_3 floor.n.01_1) 
        (onfloor cookie.n.01_4 floor.n.01_1) 
        (onfloor book.n.02_1 floor.n.01_1) 
        (onfloor book.n.02_2 floor.n.01_1) 
        (onfloor book.n.02_3 floor.n.01_1) 
        (onfloor book.n.02_4 floor.n.01_1) 
        (onfloor wrapping.n.01_1 floor.n.01_1) 
        (inroom floor.n.01_1 living_room) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forpairs 
                (?candy.n.01 - candy.n.01) 
                (?basket.n.01 - basket.n.01) 
                (inside ?candy.n.01 ?basket.n.01)
            ) 
            (forpairs 
                (?cookie.n.01 - cookie.n.01) 
                (?basket.n.01 - basket.n.01) 
                (inside ?cookie.n.01 ?basket.n.01)
            ) 
            (forpairs 
                (?book.n.02 - book.n.02) 
                (?basket.n.01 - basket.n.01) 
                (inside ?book.n.02 ?basket.n.01)
            ) 
            (forpairs 
                (?wrapping.n.01 - wrapping.n.01) 
                (?basket.n.01 - basket.n.01) 
                (inside ?wrapping.n.01 ?basket.n.01)
            )
        )
    )
)