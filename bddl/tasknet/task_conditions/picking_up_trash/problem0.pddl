(define (problem picking_up_trash_0)
    (:domain igibson)

    (:objects
        ashcan.n.01_1 - ashcan.n.01
        paper.n.01_1 paper.n.01_2 paper.n.01_3 - paper.n.01
        bag.n.01_1 bag.n.01_2 - bag.n.01
        floor.n.01_1 floor.n.01_2 - floor.n.01
        agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor ashcan.n.01_1 floor.n.01_2) 
        (onfloor paper.n.01_1 floor.n.01_2) 
        (onfloor paper.n.01_2 floor.n.01_2) 
        (onfloor paper.n.01_3 floor.n.01_1) 
        (onfloor bag.n.01_1 floor.n.01_1) 
        (onfloor bag.n.01_2 floor.n.01_1) 
        (inroom floor.n.01_2 kitchen) 
        (inroom floor.n.01_1 living_room) 
        (onfloor agent.n.01_1 floor.n.01_2)
    )
    
    (:goal 
        (forall 
            (?paper.n.01 - paper.n.01) 
            (inside ?paper.n.01 ?ashcan.n.01_1)
        ) 
        (forall 
            (?bag.n.01 - bag.n.01) 
            (inside ?bag.n.01 ?ashcan.n.01_1)
        ) 
        (forall 
            (?magazine.n.02 - magazine.n.02) 
            (inside ?magazine.n.02 ?ashcan.n.01_1)
        )
    )
)