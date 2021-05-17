(define (problem vacuuming_floors_0)
    (:domain igibson)

    (:objects
     	floor.n.01_1 - floor.n.01
    	crumb.n.03_1 crumb.n.03_2 crumb.n.03_3 crumb.n.03_4 crumb.n.03_5 crumb.n.03_6 - crumb.n.03
    	vacuum.n.04_1 - vacuum.n.04
        ashcan.n.01_1 - ashcan.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (dusty floor.n.01_1) 
        (onfloor crumb.n.03_1 floor.n.01_1) 
        (onfloor crumb.n.03_2 floor.n.01_1) 
        (onfloor crumb.n.03_3 floor.n.01_1) 
        (onfloor crumb.n.03_4 floor.n.01_1) 
        (onfloor crumb.n.03_5 floor.n.01_1) 
        (onfloor crumb.n.03_6 floor.n.01_1) 
        (onfloor vacuum.n.04_1 floor.n.01_1) 
        (onfloor ashcan.n.01_1 floor.n.01_1) 
        (inroom floor.n.01_1 bedroom) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forall 
                (?crumb.n.03 - crumb.n.03) 
                (inside crumb.n.03 ?ashcan.n.01_1)
            ) 
            (not 
                (dusty ?floor.n.01_1)
            )
        )
    )
)