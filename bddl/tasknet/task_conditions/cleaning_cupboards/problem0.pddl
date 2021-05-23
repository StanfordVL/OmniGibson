(define (problem cleaning_cupboards_0)
    (:domain igibson)

    (:objects
     	book.n.02_1 book.n.02_2 book.n.02_3 - book.n.02
    	cabinet.n.01_1 cabinet.n.01_2 cabinet.n.01_3 - cabinet.n.01
    	pen.n.01_1 - pen.n.01
    	marker.n.03_1 marker.n.03_2 - marker.n.03
    	screwdriver.n.01_1 - screwdriver.n.01
    	scrub_brush.n.01_1 - scrub_brush.n.01
    	rag.n.01_1 - rag.n.01
    	cleansing_agent.n.01_1 - cleansing_agent.n.01
    	bin.n.01_1 - bin.n.01
    	floor.n.01_1 - floor.n.01
    	bucket.n.01_1 - bucket.n.01
        bed.n.01_1 - bed.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (inside book.n.02_1 cabinet.n.01_1) 
        (inside book.n.02_2 cabinet.n.01_2) 
        (inside book.n.02_3 cabinet.n.01_2) 
        (inside pen.n.01_1 cabinet.n.01_1) 
        (inside marker.n.03_1 cabinet.n.01_2) 
        (inside marker.n.03_2 cabinet.n.01_2) 
        (inside screwdriver.n.01_1 cabinet.n.01_3) 
        (inside scrub_brush.n.01_1 cabinet.n.01_1) 
        (inside rag.n.01_1 cabinet.n.01_1) 
        (ontop cleansing_agent.n.01_1 bed.n.01_1) 
        (onfloor bin.n.01_1 floor.n.01_1) 
        (ontop bucket.n.01_1 bed.n.01_1) 
        (dusty cabinet.n.01_1) 
        (dusty cabinet.n.01_2) 
        (inroom cabinet.n.01_1 bedroom) 
        (inroom cabinet.n.01_2 bedroom) 
        (inroom cabinet.n.01_3 bedroom) 
        (inroom floor.n.01_1 bedroom) 
        (inroom bed.n.01_1 bedroom) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forall 
                (?cabinet.n.01 - cabinet.n.01) 
                (not 
                    (dusty ?cabinet.n.01)
                )
            ) 
            (forall 
                (?book.n.02 - book.n.02) 
                (forall 
                    (?cabinet.n.01 - cabinet.n.01) 
                    (not 
                        (inside ?book.n.02 ?cabinet.n.01)
                    )
                )
            )  
            (inside ?screwdriver.n.01_1 ?bin.n.01_1) 
            (forall 
                (?marker.n.03 - marker.n.03) 
                (inside ?marker.n.03 ?bucket.n.01_1)
            ) 
            (inside ?pen.n.01_1 ?bucket.n.01_1)
        )
    )
)