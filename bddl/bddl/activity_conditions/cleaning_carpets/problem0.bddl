(define (problem cleaning_carpets_0)
    (:domain igibson)

    (:objects
     	floor.n.01_1 floor.n.01_2 - floor.n.01
        hand_towel.n.01_1 - hand_towel.n.01
    	shampoo.n.01_1 - shampoo.n.01
    	cabinet.n.01_1 - cabinet.n.01
    	washer.n.03_1 - washer.n.03
    	dryer.n.01_1 - dryer.n.01
    	door.n.01_1 - door.n.01
        sink.n.01_1 - sink.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (stained floor.n.01_1) 
        (onfloor hand_towel.n.01_1 floor.n.01_2)
        (inside shampoo.n.01_1 cabinet.n.01_1) 
        (inroom floor.n.01_1 corridor) 
        (inroom floor.n.01_2 utility_room) 
        (inroom washer.n.03_1 utility_room) 
        (inroom dryer.n.01_1 utility_room) 
        (inroom door.n.01_1 corridor) 
        (inroom cabinet.n.01_1 utility_room) 
        (inroom sink.n.01_1 utility_room)
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (not 
                (stained ?floor.n.01_1)
            )
        )
    )
)