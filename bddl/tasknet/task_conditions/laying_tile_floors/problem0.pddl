(define (problem laying_tile_floors_0)
    (:domain igibson)

    (:objects
     	tile.n.01_1 tile.n.01_2 tile.n.01_3 tile.n.01_4 tile.n.01_5 tile.n.01_6 - tile.n.01
    	floor.n.01_1 - floor.n.01
    	bucket.n.01_1 bucket.n.01_2 - bucket.n.01
    	fastener.n.02_1 fastener.n.02_2 fastener.n.02_3 - fastener.n.02
    	trowel.n.01_1 - trowel.n.01
    	saw.n.02_1 - saw.n.02
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor tile.n.01_1 floor.n.01_1) 
        (onfloor tile.n.01_2 floor.n.01_1) 
        (onfloor tile.n.01_3 floor.n.01_1) 
        (onfloor tile.n.01_4 floor.n.01_1) 
        (onfloor tile.n.01_5 floor.n.01_1) 
        (onfloor tile.n.01_6 floor.n.01_1) 
        (onfloor bucket.n.01_1 floor.n.01_1) 
        (onfloor bucket.n.01_2 floor.n.01_1) 
        (inside fastener.n.02_1 bucket.n.01_1) 
        (inside fastener.n.02_2 bucket.n.01_1) 
        (inside fastener.n.02_3 bucket.n.01_2) 
        (onfloor trowel.n.01_1 floor.n.01_1) 
        (onfloor saw.n.02_1 floor.n.01_1) 
        (inroom floor.n.01_1 corridor) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forall 
                (?fastener.n.02 - fastener.n.02) 
                (exists 
                    (?tile.n.01 - tile.n.01) 
                    (under ?fastener.n.02 ?tile.n.01)
                )
            )
        )
    )
)