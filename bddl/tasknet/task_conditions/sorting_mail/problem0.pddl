(define (problem sorting_mail_0)
    (:domain igibson)

    (:objects
     	envelope.n.01_1 envelope.n.01_2 envelope.n.01_3 envelope.n.01_4 - envelope.n.01
    	floor.n.01_1 - floor.n.01
    	sofa.n.01_1 - sofa.n.01
    	package.n.02_1 package.n.02_2 package.n.02_3 package.n.02_4 - package.n.02
    	table.n.02_1 - table.n.02
        agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor envelope.n.01_1 floor.n.01_1) 
        (ontop envelope.n.01_2 sofa.n.01_1) 
        (ontop envelope.n.01_3 sofa.n.01_1) 
        (onfloor envelope.n.01_4 floor.n.01_1) 
        (onfloor package.n.02_1 floor.n.01_1) 
        (onfloor package.n.02_2 floor.n.01_1) 
        (onfloor package.n.02_3 floor.n.01_1) 
        (onfloor package.n.02_4 floor.n.01_1) 
        (inroom floor.n.01_1 living_room) 
        (inroom sofa.n.01_1 living_room) 
        (inroom table.n.02_1 living_room) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (and 
                (forall 
                    (?package.n.02 - package.n.02) 
                    (onfloor ?package.n.02 ?floor.n.01_1)
                ) 
                (forall 
                    (?envelope.n.01 - envelope.n.01) 
                    (ontop ?envelope.n.01 ?sofa.n.01_1)
                )
            )
        )
    )
)