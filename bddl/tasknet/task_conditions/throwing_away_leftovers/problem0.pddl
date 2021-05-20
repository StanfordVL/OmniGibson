(define (problem throwing_away_leftovers_0)
    (:domain igibson)

    (:objects
     	plate.n.04_1 plate.n.04_2 plate.n.04_3 plate.n.04_4 - plate.n.04
        hamburger.n.01_1 hamburger.n.01_2 hamburger.n.01_3 - hamburger.n.01
    	sofa.n.01_1 - sofa.n.01
    	floor.n.01_1 - floor.n.01
    	ashcan.n.01_1 - ashcan.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (ontop plate.n.04_1 sofa.n.01_1) 
        (ontop hamburger.n.01_1 plate.n.04_1) 
        (ontop plate.n.04_2 sofa.n.01_1) 
        (ontop hamburger.n.01_3 plate.n.04_2) 
        (ontop plate.n.04_3 sofa.n.01_1) 
        (ontop hamburger.n.01_2 plate.n.04_3) 
        (ontop plate.n.04_4 sofa.n.01_1) 
        (inroom sofa.n.01_1 living_room) 
        (inroom floor.n.01_1 living_room) 
        (onfloor ashcan.n.01_1 floor.n.01_1) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forall 
                (?hamburger.n.01 - hamburger.n.01) 
                (inside ?hamburger.n.01_1 ?ashcan.n.01_1)
            )
        )
    )
)