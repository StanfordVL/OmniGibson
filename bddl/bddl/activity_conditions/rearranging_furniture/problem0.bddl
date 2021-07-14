(define (problem rearranging_furniture_0)
    (:domain igibson)

    (:objects
     	lamp.n.02_1 lamp.n.02_2 - lamp.n.02
    	floor.n.01_1 - floor.n.01
    	seat.n.03_1 seat.n.03_2 - seat.n.03
    	bed.n.01_1 - bed.n.01
    	window.n.01_1 - window.n.01
    	door.n.01_1 - door.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor lamp.n.02_1 floor.n.01_1) 
        (onfloor lamp.n.02_2 floor.n.01_1) 
        (onfloor seat.n.03_1 floor.n.01_1) 
        (ontop seat.n.03_2 bed.n.01_1) 
        (inroom window.n.01_1 bedroom) 
        (inroom bed.n.01_1 bedroom) 
        (inroom floor.n.01_1 bedroom) 
        (inroom door.n.01_1 bedroom) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (nextto ?lamp.n.02_1 ?door.n.01_1) 
            (nextto ?lamp.n.02_2 ?window.n.01_1) 
            (touching ?seat.n.03_1 ?bed.n.01_1) 
            (nextto ?seat.n.03_2 ?window.n.01_1)
        )
    )
)