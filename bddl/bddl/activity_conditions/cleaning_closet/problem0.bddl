(define (problem cleaning_closet_0)
    (:domain igibson)

    (:objects
     	shelf.n.01_1 - shelf.n.01
    	cabinet.n.01_1 cabinet.n.01_2 cabinet.n.01_3 - cabinet.n.01
    	jewelry.n.01_1 jewelry.n.01_2 - jewelry.n.01
    	floor.n.01_1 floor.n.01_2 - floor.n.01
    	hat.n.01_1 - hat.n.01
    	sandal.n.01_1 sandal.n.01_2 - sandal.n.01
    	umbrella.n.01_1 - umbrella.n.01
    	towel.n.01_1 - towel.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (dusty shelf.n.01_1) 
        (dusty cabinet.n.01_3) 
        (onfloor jewelry.n.01_1 floor.n.01_1) 
        (onfloor jewelry.n.01_2 floor.n.01_1) 
        (dusty floor.n.01_1) 
        (onfloor hat.n.01_1 floor.n.01_1) 
        (onfloor sandal.n.01_1 floor.n.01_1) 
        (onfloor sandal.n.01_2 floor.n.01_1) 
        (onfloor umbrella.n.01_1 floor.n.01_1) 
        (inside towel.n.01_1 cabinet.n.01_1) 
        (inroom cabinet.n.01_1 bedroom) 
        (inroom cabinet.n.01_2 bedroom) 
        (inroom cabinet.n.01_3 closet) 
        (inroom shelf.n.01_1 closet) 
        (inroom floor.n.01_1 closet) 
        (inroom floor.n.01_2 bedroom) 
        (onfloor agent.n.01_1 floor.n.01_2)
    )
    
    (:goal 
        (and 
            (forall 
                (?jewelry.n.01 - jewelry.n.01) 
                (inside ?jewelry.n.01 ?cabinet.n.01_3)
            ) 
            (not 
                (inside ?umbrella.n.01_1 ?cabinet.n.01_3)
            ) 
            (or 
                (inside ?hat.n.01_1 ?cabinet.n.01_1) 
                (ontop ?hat.n.01_1 ?shelf.n.01_1)
            ) 
            (forall 
                (?sandal.n.01 - sandal.n.01) 
                (and 
                    (nextto ?sandal.n.01 ?shelf.n.01_1) 
                    (onfloor ?sandal.n.01 ?floor.n.01_1)
                )
            ) 
            (not 
                (dusty ?cabinet.n.01_3)
            ) 
            (not 
                (dusty ?shelf.n.01_1)
            ) 
            (not 
                (dusty ?floor.n.01_1)
            )
        )
    )
)