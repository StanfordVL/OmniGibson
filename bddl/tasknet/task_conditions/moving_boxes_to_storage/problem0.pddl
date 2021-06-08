(define (problem moving_boxes_to_storage_0)
    (:domain igibson)

    (:objects
        carton.n.02_1 carton.n.02_2 - carton.n.02
    	floor.n.01_1 floor.n.01_2 - floor.n.01
    	shelf.n.01_1 - shelf.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor carton.n.02_1 floor.n.01_1)
        (onfloor carton.n.02_2 floor.n.01_1)
        (inroom floor.n.01_1 living_room) 
        (inroom floor.n.01_2 storage_room) 
        (inroom shelf.n.01_1 living_room) 
        (onfloor agent.n.01_1 floor.n.01_2)
    )
    
    (:goal 
        (and 
            (onfloor ?carton.n.02_1 ?floor.n.01_2)
            (ontop ?carton.n.02_2 ?carton.n.02_1)
        )
    )
)