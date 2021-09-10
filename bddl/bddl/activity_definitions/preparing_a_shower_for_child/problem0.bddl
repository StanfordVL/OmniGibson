(define (problem preparing_a_shower_for_child_0)
    (:domain igibson)

    (:objects
     	shampoo.n.01_1 - shampoo.n.01
    	floor.n.01_1 - floor.n.01
    	soap.n.01_1 - soap.n.01
    	towel.n.01_1 - towel.n.01
    	shower.n.01_1 - shower.n.01
    	sink.n.01_1 - sink.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor shampoo.n.01_1 floor.n.01_1) 
        (onfloor soap.n.01_1 floor.n.01_1) 
        (onfloor towel.n.01_1 floor.n.01_1) 
        (inroom shower.n.01_1 bathroom) 
        (inroom floor.n.01_1 bathroom) 
        (inroom sink.n.01_1 bathroom) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (onfloor ?towel.n.01_1 ?floor.n.01_1) 
            (onfloor ?shampoo.n.01_1 ?floor.n.01_1) 
            (nextto ?soap.n.01_1 ?sink.n.01_1)
        )
    )
)