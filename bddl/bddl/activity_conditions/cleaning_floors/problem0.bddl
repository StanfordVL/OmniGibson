(define (problem cleaning_floors_0)
    (:domain igibson)

    (:objects
     	floor.n.01_1 - floor.n.01
    	broom.n.01_1 - broom.n.01
    	dustpan.n.02_1 - dustpan.n.02
    	cleansing_agent.n.01_1 - cleansing_agent.n.01
    	scrub_brush.n.01_1 - scrub_brush.n.01
    	door.n.01_1 - door.n.01
    	sink.n.01_1 - sink.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (dusty floor.n.01_1) 
        (stained floor.n.01_1) 
        (onfloor broom.n.01_1 floor.n.01_1) 
        (onfloor dustpan.n.02_1 floor.n.01_1) 
        (onfloor cleansing_agent.n.01_1 floor.n.01_1) 
        (onfloor scrub_brush.n.01_1 floor.n.01_1) 
        (inroom floor.n.01_1 bathroom) 
        (inroom door.n.01_1 bathroom) 
        (inroom sink.n.01_1 bathroom) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (not 
                (stained ?floor.n.01_1)
            ) 
            (not 
                (dusty ?floor.n.01_1)
            )
        )
    )
)