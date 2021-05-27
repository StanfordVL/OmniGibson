(define (problem cleaning_toilet_0)
    (:domain igibson)

    (:objects
     	toilet.n.02_1 - toilet.n.02
    	scrub_brush.n.01_1 - scrub_brush.n.01
    	floor.n.01_1 - floor.n.01
    	detergent.n.02_1 - detergent.n.02
        sink.n.01_1 - sink.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (stained toilet.n.02_1) 
        (onfloor scrub_brush.n.01_1 floor.n.01_1) 
        (onfloor detergent.n.02_1 floor.n.01_1) 
        (inroom toilet.n.02_1 bathroom) 
        (inroom floor.n.01_1 bathroom) 
        (inroom sink.n.01_1 bathroom)
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (not 
                (stained ?toilet.n.02_1)
            ) 
            (onfloor ?scrub_brush.n.01_1 ?floor.n.01_1) 
            (onfloor ?detergent.n.02_1 ?floor.n.01_1)
        )
    )
)