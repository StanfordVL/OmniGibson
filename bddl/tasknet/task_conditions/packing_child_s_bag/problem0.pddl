(define (problem packing_child_s_bag_0)
    (:domain igibson)

    (:objects
     	duffel_bag.n.01_1 - duffel_bag.n.01
    	floor.n.01_1 - floor.n.01
    	towel.n.01_1 - towel.n.01
    	bed.n.01_1 - bed.n.01
    	hat.n.01_1 - hat.n.01
    	sunglass.n.01_1 - sunglass.n.01
    	sunscreen.n.01_1 - sunscreen.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor duffel_bag.n.01_1 floor.n.01_1) 
        (ontop towel.n.01_1 bed.n.01_1) 
        (ontop hat.n.01_1 bed.n.01_1) 
        (ontop sunglass.n.01_1 bed.n.01_1) 
        (onfloor sunscreen.n.01_1 floor.n.01_1) 
        (inroom bed.n.01_1 bedroom) 
        (inroom floor.n.01_1 bedroom) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (nextto ?duffel_bag.n.01_1 ?bed.n.01_1) 
            (inside ?towel.n.01_1 ?duffel_bag.n.01_1) 
            (inside ?sunglass.n.01_1 ?duffel_bag.n.01_1) 
            (inside ?sunscreen.n.01_1 ?duffel_bag.n.01_1) 
            (ontop ?hat.n.01_1 ?duffel_bag.n.01_1)
        )
    )
)