(define (problem cleaning_table_after_clearing_0)
    (:domain igibson)

    (:objects
     	table.n.02_1 - table.n.02
    	soap.n.01_1 - soap.n.01
    	sink.n.01_1 - sink.n.01
    	floor.n.01_1 - floor.n.01
            dishtowel.n.01_1 - dishtowel.n.01
            cabinet.n.01_1 - cabinet.n.01
            agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (stained table.n.02_1) 
        (inside soap.n.01_1 cabinet.n.01_1) 
        (inside dishtowel.n.01_1 cabinet.n.01_1) 
        (inroom table.n.02_1 dining_room) 
        (inroom floor.n.01_1 dining_room) 
        (inroom cabinet.n.01_1 kitchen) 
        (inroom sink.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (not 
                (stained ?table.n.02_1)
            )
        )
    )
)