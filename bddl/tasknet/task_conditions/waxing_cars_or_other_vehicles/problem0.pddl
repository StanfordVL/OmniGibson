(define (problem waxing_cars_or_other_vehicles_0)
    (:domain igibson)

    (:objects
     	vehicle.n.01_1 - vehicle.n.01
    	floor.n.01_1 - floor.n.01
    	vessel.n.03_1 - vessel.n.03
    	shelf.n.01_1 - shelf.n.01
    	applicator.n.01_1 - applicator.n.01
    	pad.n.04_1 - pad.n.04
        agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor vehicle.n.01_1 floor.n.01_1) 
        (dusty vehicle.n.01_1) 
        (ontop vessel.n.03_1 shelf.n.01_1) 
        (ontop applicator.n.01_1 shelf.n.01_1) 
        (inside pad.n.04_1 shelf.n.01_1) 
        (inroom floor.n.01_1 garage) 
        (inroom shelf.n.01_1 garage) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (not 
                (dusty ?vehicle.n.01_1)
            ) 
            (nextto ?pad.n.04_1 ?vessel.n.03_1)
        )
    )
)