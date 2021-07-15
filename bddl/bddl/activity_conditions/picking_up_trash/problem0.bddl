(define (problem picking_up_trash_0)
    (:domain igibson)

    (:objects
        ashcan.n.01_1 - ashcan.n.01
        pad.n.01_1 pad.n.01_2 pad.n.01_3 - pad.n.01
        pop.n.02_1 pop.n.02_2 - pop.n.02
        floor.n.01_1 floor.n.01_2 - floor.n.01
        agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor ashcan.n.01_1 floor.n.01_2) 
        (onfloor pad.n.01_1 floor.n.01_2) 
        (onfloor pad.n.01_2 floor.n.01_2) 
        (onfloor pad.n.01_3 floor.n.01_1) 
        (onfloor pop.n.02_1 floor.n.01_1) 
        (onfloor pop.n.02_2 floor.n.01_1) 
        (inroom floor.n.01_2 kitchen) 
        (inroom floor.n.01_1 living_room) 
        (onfloor agent.n.01_1 floor.n.01_2)
    )
    
    (:goal 
        (and 
            (forall 
                (?pad.n.01 - pad.n.01) 
                (inside ?pad.n.01 ?ashcan.n.01_1)
            ) 
            (forall 
                (?pop.n.02 - pop.n.02) 
                (inside ?pop.n.02 ?ashcan.n.01_1)
            )
        )
    )
)