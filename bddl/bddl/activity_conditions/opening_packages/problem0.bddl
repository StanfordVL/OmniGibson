(define (problem opening_packages_0)
    (:domain igibson)

    (:objects
     	package.n.02_1 package.n.02_2 - package.n.02
    	floor.n.01_1 - floor.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor package.n.02_1 floor.n.01_1) 
        (onfloor package.n.02_2 floor.n.01_1) 
        (not 
            (open package.n.02_1)
        ) 
        (not 
            (open package.n.02_2)
        ) 
        (inroom floor.n.01_1 bedroom) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forall 
                (?package.n.02 - package.n.02) 
                (open ?package.n.02)
            )
        )
    )
)