(define (problem cleaning_the_hot_tub_0)
    (:domain igibson)

    (:objects
     	pool.n.01_1 - pool.n.01
    	floor.n.01_1 - floor.n.01
    	scrub_brush.n.01_1 - scrub_brush.n.01
        sink.n.01_1 - sink.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor pool.n.01_1 floor.n.01_1) 
        (stained pool.n.01_1) 
        (onfloor scrub_brush.n.01_1 floor.n.01_1) 
        (inroom floor.n.01_1 garage) 
        (inroom sink.n.01_1 bathroom)
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (onfloor ?pool.n.01_1 ?floor.n.01_1) 
            (not 
                (stained ?pool.n.01_1)
            )
        )
    )
)