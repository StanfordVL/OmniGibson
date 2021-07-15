(define (problem cleaning_shoes_0)
    (:domain igibson)

    (:objects
     	soap.n.01_1 - soap.n.01
    	bed.n.01_1 - bed.n.01
    	floor.n.01_1 - floor.n.01
    	rag.n.01_1 - rag.n.01
    	towel.n.01_1 - towel.n.01
    	shoe.n.01_1 shoe.n.01_2 shoe.n.01_3 shoe.n.01_4 - shoe.n.01
        sink.n.01_1 - sink.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (ontop soap.n.01_1 bed.n.01_1) 
        (ontop rag.n.01_1 bed.n.01_1) 
        (onfloor towel.n.01_1 floor.n.01_1) 
        (ontop shoe.n.01_1 bed.n.01_1) 
        (ontop shoe.n.01_2 bed.n.01_1) 
        (ontop shoe.n.01_3 bed.n.01_1) 
        (ontop shoe.n.01_4 bed.n.01_1) 
        (stained shoe.n.01_1) 
        (stained shoe.n.01_2) 
        (dusty shoe.n.01_3) 
        (dusty shoe.n.01_4) 
        (not 
            (soaked rag.n.01_1)
        ) 
        (inroom floor.n.01_1 bedroom) 
        (inroom bed.n.01_1 bedroom) 
        (inroom sink.n.01_1 bathroom)
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forall 
                (?shoe.n.01 - shoe.n.01) 
                (not 
                    (stained ?shoe.n.01)
                )
            ) 
            (forall 
                (?shoe.n.01 - shoe.n.01) 
                (not 
                    (dusty ?shoe.n.01)
                )
            ) 
            (onfloor ?towel.n.01_1 ?floor.n.01_1)
        )
    )
)