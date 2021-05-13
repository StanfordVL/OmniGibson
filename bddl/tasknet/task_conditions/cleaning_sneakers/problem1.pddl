(define (problem cleaning_sneakers_1)
    (:domain igibson)

    (:objects
     	gym_shoe.n.01_1 gym_shoe.n.01_2 - gym_shoe.n.01
    	floor.n.01_1 - floor.n.01
    	shoe.n.01_1 shoe.n.01_2 - shoe.n.01
    	soap.n.01_1 - soap.n.01
    	brush.n.02_1 - brush.n.02
    	sink.n.01_1 - sink.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor gym_shoe.n.01_1 floor.n.01_1) 
        (stained gym_shoe.n.01_1) 
        (onfloor gym_shoe.n.01_2 floor.n.01_1) 
        (stained gym_shoe.n.01_2) 
        (onfloor shoe.n.01_1 floor.n.01_1) 
        (stained shoe.n.01_1) 
        (onfloor shoe.n.01_2 floor.n.01_1) 
        (stained shoe.n.01_2) 
        (onfloor soap.n.01_1 floor.n.01_1) 
        (onfloor brush.n.02_1 floor.n.01_1) 
        (inroom sink.n.01_1 bathroom) 
        (inroom floor.n.01_1 bathroom) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forall 
                (?gym_shoe.n.01 - gym_shoe.n.01) 
                (not 
                    (stained ?gym_shoe.n.01)
                )
            ) 
            (forall 
                (?shoe.n.01 - shoe.n.01) 
                (not 
                    (stained ?shoe.n.01)
                )
            ) 
            (nextto ?brush.n.02_1 ?sink.n.01_1)
        )
    )
)