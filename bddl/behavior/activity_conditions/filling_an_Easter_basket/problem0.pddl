(define (problem filling_an_Easter_basket_0)
    (:domain igibson)

    (:objects
     	basket.n.01_1 basket.n.01_2 - basket.n.01
    	countertop.n.01_1 - countertop.n.01
    	ball.n.01_1 - ball.n.01
    	jewelry.n.01_1 - jewelry.n.01
    	book.n.02_1 book.n.02_2 - book.n.02
    	cabinet.n.01_1 cabinet.n.01_2 - cabinet.n.01
    	bow.n.08_1 bow.n.08_2 - bow.n.08
    	egg.n.02_1 egg.n.02_2 - egg.n.02
    	electric_refrigerator.n.01_1 - electric_refrigerator.n.01
    	candy.n.01_1 candy.n.01_2 - candy.n.01
    	floor.n.01_1 - floor.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (ontop basket.n.01_1 countertop.n.01_1) 
        (ontop basket.n.01_2 countertop.n.01_1) 
        (ontop ball.n.01_1 countertop.n.01_1) 
        (ontop jewelry.n.01_1 countertop.n.01_1) 
        (inside book.n.02_1 cabinet.n.01_1) 
        (inside book.n.02_2 cabinet.n.01_1) 
        (inside bow.n.08_1 cabinet.n.01_2) 
        (inside bow.n.08_2 cabinet.n.01_2) 
        (inside egg.n.02_1 electric_refrigerator.n.01_1) 
        (cooked egg.n.02_1) 
        (inside egg.n.02_2 electric_refrigerator.n.01_1) 
        (cooked egg.n.02_2) 
        (ontop candy.n.01_1 electric_refrigerator.n.01_1) 
        (ontop candy.n.01_2 electric_refrigerator.n.01_1) 
        (inroom countertop.n.01_1 kitchen) 
        (inroom cabinet.n.01_1 kitchen) 
        (inroom cabinet.n.01_2 kitchen) 
        (inroom electric_refrigerator.n.01_1 kitchen) 
        (inroom floor.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forall 
                (?basket.n.01 - basket.n.01) 
                (ontop ?basket.n.01 ?countertop.n.01_1)
            ) 
            (forpairs 
                (?basket.n.01 - basket.n.01) 
                (?egg.n.02 - egg.n.02) 
                (inside ?egg.n.02 ?basket.n.01)
            ) 
            (forpairs 
                (?basket.n.01 - basket.n.01) 
                (?candy.n.01 - candy.n.01) 
                (inside ?candy.n.01 ?basket.n.01)
            ) 
            (exists 
                (?basket.n.01 - basket.n.01) 
                (and 
                    (inside ?jewelry.n.01_1 ?basket.n.01) 
                    (inside ?ball.n.01_1 ?basket.n.01)
                )
            ) 
            (forpairs 
                (?bow.n.08 - bow.n.08) 
                (?basket.n.01 - basket.n.01) 
                (or 
                    (ontop ?bow.n.08 ?basket.n.01) 
                    (inside ?bow.n.08 ?basket.n.01)
                )
            ) 
            (forpairs 
                (?book.n.02 - book.n.02) 
                (?basket.n.01 - basket.n.01) 
                (nextto ?book.n.02 ?basket.n.01)
            )
        )
    )
)