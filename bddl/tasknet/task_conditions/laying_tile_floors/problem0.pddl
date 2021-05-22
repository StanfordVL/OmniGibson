(define (problem laying_tile_floors_0)
    (:domain igibson)

    (:objects
        tile.n.01_1 tile.n.01_2 tile.n.01_3 tile.n.01_4 - tile.n.01
        floor.n.01_1 floor.n.01_2 - floor.n.01
        agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor tile.n.01_1 floor.n.01_1) 
        (onfloor tile.n.01_2 floor.n.01_1) 
        (onfloor tile.n.01_3 floor.n.01_1) 
        (onfloor tile.n.01_4 floor.n.01_1) 
        (inroom floor.n.01_1 corridor) 
        (inroom floor.n.01_2 bathroom) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forall 
                (?tile.n.01 - tile.n.01) 
                (onfloor ?tile.n.01 ?floor.n.01_2)
            )
        )
    )
)