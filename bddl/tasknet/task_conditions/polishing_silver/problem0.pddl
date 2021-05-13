(define (problem polishing_silver_0)
    (:domain igibson)

    (:objects
     	cutlery.n.02_1 cutlery.n.02_2 cutlery.n.02_3 cutlery.n.02_4 - cutlery.n.02
    	piece_of_cloth.n.01_1 - piece_of_cloth.n.01
    	cabinet.n.01_1 - cabinet.n.01
    	floor.n.01_1 - floor.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (dusty cutlery.n.02_1) 
        (dusty cutlery.n.02_2) 
        (dusty cutlery.n.02_3) 
        (dusty cutlery.n.02_4) 
        (inside piece_of_cloth.n.01_1 cabinet.n.01_1) 
        (inside cutlery.n.02_1 cabinet.n.01_1) 
        (inside cutlery.n.02_2 cabinet.n.01_1) 
        (inside cutlery.n.02_3 cabinet.n.01_1) 
        (inside cutlery.n.02_4 cabinet.n.01_1) 
        (inroom cabinet.n.01_1 kitchen) 
        (inroom floor.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (not 
                (dusty ?cutlery.n.02_1)
            ) 
            (not 
                (dusty ?cutlery.n.02_2)
            ) 
            (not 
                (dusty ?cutlery.n.02_3)
            ) 
            (not 
                (dusty ?cutlery.n.02_4)
            ) 
            (not 
                (inside ?piece_of_cloth.n.01_1 ?cabinet.n.01_1)
            ) 
            (inside ?cutlery.n.02_1 ?cabinet.n.01_1) 
            (inside ?cutlery.n.02_2 ?cabinet.n.01_1) 
            (inside ?cutlery.n.02_3 ?cabinet.n.01_1) 
            (inside ?cutlery.n.02_4 ?cabinet.n.01_1)
        )
    )
)