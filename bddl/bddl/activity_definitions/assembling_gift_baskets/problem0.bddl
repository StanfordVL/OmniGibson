(define (problem assembling_gift_baskets_0)
    (:domain igibson)

    (:objects
     	basket.n.01_1 basket.n.01_2 basket.n.01_3 basket.n.01_4 - basket.n.01
    	floor.n.01_1 - floor.n.01
    	candle.n.01_1 candle.n.01_2 candle.n.01_3 candle.n.01_4 - candle.n.01
    	cookie.n.01_1 cookie.n.01_2 cookie.n.01_3 cookie.n.01_4 - cookie.n.01
    	cheese.n.01_1 cheese.n.01_2 cheese.n.01_3 cheese.n.01_4 - cheese.n.01
    	bow.n.08_1 bow.n.08_2 bow.n.08_3 bow.n.08_4 - bow.n.08
        table.n.02_1 table.n.02_2 - table.n.02
        agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor basket.n.01_1 floor.n.01_1) 
        (onfloor basket.n.01_2 floor.n.01_1) 
        (onfloor basket.n.01_3 floor.n.01_1) 
        (onfloor basket.n.01_4 floor.n.01_1) 
        (ontop candle.n.01_1 table.n.02_1) 
        (ontop candle.n.01_2 table.n.02_1) 
        (ontop candle.n.01_3 table.n.02_1) 
        (ontop candle.n.01_4 table.n.02_1) 
        (ontop cookie.n.01_1 table.n.02_1) 
        (ontop cookie.n.01_2 table.n.02_1) 
        (ontop cookie.n.01_3 table.n.02_1) 
        (ontop cookie.n.01_4 table.n.02_1) 
        (ontop cheese.n.01_1 table.n.02_2) 
        (ontop cheese.n.01_2 table.n.02_2) 
        (ontop cheese.n.01_3 table.n.02_2) 
        (ontop cheese.n.01_4 table.n.02_2) 
        (ontop bow.n.08_1 table.n.02_2) 
        (ontop bow.n.08_2 table.n.02_2) 
        (ontop bow.n.08_3 table.n.02_2) 
        (ontop bow.n.08_4 table.n.02_2) 
        (inroom floor.n.01_1 living_room) 
        (inroom table.n.02_1 living_room) 
        (inroom table.n.02_2 living_room) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forpairs 
                (?basket.n.01 - basket.n.01) 
                (?candle.n.01 - candle.n.01) 
                (inside ?candle.n.01 ?basket.n.01)
            ) 
            (forpairs 
                (?basket.n.01 - basket.n.01) 
                (?cheese.n.01 - cheese.n.01) 
                (inside ?cheese.n.01 ?basket.n.01)
            ) 
            (forpairs 
                (?basket.n.01 - basket.n.01) 
                (?cookie.n.01 - cookie.n.01) 
                (inside ?cookie.n.01 ?basket.n.01)
            ) 
            (forpairs 
                (?basket.n.01 - basket.n.01) 
                (?bow.n.08 - bow.n.08) 
                (inside ?bow.n.08 ?basket.n.01)
            )
        )
    )
)