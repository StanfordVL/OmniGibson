(define (problem bringing_in_wood_0)
    (:domain igibson)

    (:objects
        plywood.n.01_1 plywood.n.01_2 plywood.n.01_3 - plywood.n.01
        floor.n.01_1 floor.n.01_2 - floor.n.01
        agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor plywood.n.01_1 floor.n.01_1) 
        (onfloor plywood.n.01_2 floor.n.01_1) 
        (onfloor plywood.n.01_3 floor.n.01_1) 
        (inroom floor.n.01_1 living_room) 
        (inroom floor.n.01_2 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forall 
                (?plywood.n.01 - plywood.n.01) 
                (onfloor ?plywood.n.01 ?floor.n.01_2)
            )
        )
    )
)