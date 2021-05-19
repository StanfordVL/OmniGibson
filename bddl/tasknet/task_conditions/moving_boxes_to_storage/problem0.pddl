(define (problem moving_boxes_to_storage_0)
    (:domain igibson)

    (:objects
     	box.n.01_1 box.n.01_2 box.n.01_3 box.n.01_4 - box.n.01
    	floor.n.01_1 floor.n.01_2 - floor.n.01
    	shelf.n.01_1 - shelf.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor box.n.01_1 floor.n.01_1) 
        (onfloor box.n.01_2 floor.n.01_1) 
        (onfloor box.n.01_3 floor.n.01_1) 
        (onfloor box.n.01_4 floor.n.01_1) 
        (inroom floor.n.01_1 living_room) 
        (inroom floor.n.01_2 storage_room) 
        (inroom shelf.n.01_1 living_room) 
        (onfloor agent.n.01_1 floor.n.01_2)
    )
    
    (:goal 
        (and 
            (onfloor ?box.n.01_1 ?floor.n.01_2) 
            (ontop ?box.n.01_2 ?box.n.01_1) 
            (ontop ?box.n.01_3 ?box.n.01_2) 
            (onfloor ?box.n.01_4 ?floor.n.01_2)
        )
    )
)