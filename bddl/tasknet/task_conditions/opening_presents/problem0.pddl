(define (problem opening_presents_0)
    (:domain igibson)

    (:objects
     	package.n.02_1 package.n.02_2 package.n.02_3 package.n.02_4 - package.n.02
    	floor.n.01_1 - floor.n.01
    	rug.n.01_1 - rug.n.01
    	sofa.n.01_1 - sofa.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor package.n.02_1 floor.n.01_1) 
        (onfloor package.n.02_2 floor.n.01_1) 
        (onfloor package.n.02_3 floor.n.01_1) 
        (ontop package.n.02_4 rug.n.01_1) 
        (not 
            (open package.n.02_1)
        ) 
        (not 
            (open package.n.02_2)
        ) 
        (not 
            (open package.n.02_3)
        ) 
        (not 
            (open package.n.02_4)
        ) 
        (inroom floor.n.01_1 living_room) 
        (inroom sofa.n.01_1 living_room) 
        (inroom rug.n.01_1 living_room) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forall 
                (?package.n.02 - package.n.02) 
                (and 
                    (onfloor ?package.n.02 ?floor.n.01_1) 
                    (open ?package.n.02)
                )
            )
        )
    )
)