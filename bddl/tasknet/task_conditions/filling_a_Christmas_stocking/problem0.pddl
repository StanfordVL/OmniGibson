(define (problem filling_a_Christmas_stocking_0)
    (:domain igibson)

    (:objects
     	cube.n.05_1 cube.n.05_2 cube.n.05_3 cube.n.05_4 - cube.n.05
    	floor.n.01_1 - floor.n.01
    	candy.n.01_1 candy.n.01_2 candy.n.01_3 candy.n.01_4 - candy.n.01
    	cabinet.n.01_1 cabinet.n.01_2 - cabinet.n.01
    	pen.n.01_1 pen.n.01_2 pen.n.01_3 pen.n.01_4 - pen.n.01
    	stocking.n.01_1 stocking.n.01_2 stocking.n.01_3 stocking.n.01_4 - stocking.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor cube.n.05_1 floor.n.01_1) 
        (onfloor cube.n.05_2 floor.n.01_1) 
        (onfloor cube.n.05_3 floor.n.01_1) 
        (onfloor cube.n.05_4 floor.n.01_1) 
        (inside candy.n.01_1 cabinet.n.01_1) 
        (inside candy.n.01_2 cabinet.n.01_1) 
        (inside candy.n.01_3 cabinet.n.01_1) 
        (inside candy.n.01_4 cabinet.n.01_1) 
        (inside pen.n.01_1 cabinet.n.01_1) 
        (inside pen.n.01_2 cabinet.n.01_1) 
        (inside pen.n.01_3 cabinet.n.01_1) 
        (inside pen.n.01_4 cabinet.n.01_1) 
        (onfloor stocking.n.01_1 floor.n.01_1) 
        (onfloor stocking.n.01_2 floor.n.01_1) 
        (onfloor stocking.n.01_3 floor.n.01_1) 
        (onfloor stocking.n.01_4 floor.n.01_1) 
        (inroom floor.n.01_1 living_room) 
        (inroom cabinet.n.01_1 kitchen) 
        (inroom cabinet.n.01_2 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forpairs 
                (?stocking.n.01 - stocking.n.01) 
                (?cube.n.05 - cube.n.05) 
                (inside ?cube.n.05 ?stocking.n.01)
            ) 
            (forpairs 
                (?stocking.n.01 - stocking.n.01) 
                (?candy.n.01 - candy.n.01) 
                (inside ?candy.n.01 ?stocking.n.01)
            ) 
            (forpairs 
                (?stocking.n.01 - stocking.n.01) 
                (?pen.n.01 - pen.n.01) 
                (inside ?pen.n.01 ?stocking.n.01)
            )
        )
    )
)