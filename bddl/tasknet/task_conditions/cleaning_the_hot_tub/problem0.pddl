(define (problem cleaning_the_hot_tub_0)
    (:domain igibson)

    (:objects
     	pool.n.01_1 - pool.n.01
    	floor.n.01_1 - floor.n.01
    	detergent.n.02_1 - detergent.n.02
    	brush.n.02_1 - brush.n.02
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor pool.n.01_1 floor.n.01_1) 
        (stained pool.n.01_1) 
        (onfloor detergent.n.02_1 floor.n.01_1) 
        (onfloor brush.n.02_1 floor.n.01_1) 
        (inroom floor.n.01_1 corridor) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (onfloor ?pool.n.01_1 ?floor.n.01_1) 
            (not 
                (stained ?pool.n.01_1)
            ) 
            (inside ?brush.n.02_1 ?pool.n.01_1) 
            (inside ?detergent.n.02_1 ?pool.n.01_1)
        )
    )
)