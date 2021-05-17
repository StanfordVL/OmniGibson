(define (problem cleaning_windows_0)
    (:domain igibson)

    (:objects
     	towel.n.01_1 towel.n.01_2 - towel.n.01
    	cabinet.n.01_1 - cabinet.n.01
    	rag.n.01_1 rag.n.01_2 - rag.n.01
    	cleansing_agent.n.01_1 - cleansing_agent.n.01
    	window.n.01_1 window.n.01_2 - window.n.01
    	sink.n.01_1 - sink.n.01
    	floor.n.01_1 floor.n.01_2 - floor.n.01
    	table.n.02_1 - table.n.02
        agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (inside towel.n.01_1 cabinet.n.01_1) 
        (inside towel.n.01_2 cabinet.n.01_1) 
        (inside rag.n.01_1 cabinet.n.01_1) 
        (inside rag.n.01_2 cabinet.n.01_1) 
        (not 
            (soaked rag.n.01_1)
        ) 
        (not 
            (soaked rag.n.01_2)
        ) 
        (inside cleansing_agent.n.01_1 cabinet.n.01_1) 
        (dusty window.n.01_1) 
        (dusty window.n.01_2) 
        (not 
            (dusty sink.n.01_1)
        ) 
        (inroom floor.n.01_1 kitchen) 
        (inroom floor.n.01_2 living_room) 
        (inroom window.n.01_1 kitchen) 
        (inroom window.n.01_2 living_room) 
        (inroom cabinet.n.01_1 kitchen) 
        (inroom sink.n.01_1 kitchen) 
        (inroom table.n.02_1 living_room) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (and 
                (soaked ?rag.n.01_1) 
                (soaked ?rag.n.01_2)
            ) 
            (and 
                (not 
                    (dusty ?window.n.01_1)
                ) 
                (not 
                    (dusty ?window.n.01_2)
                )
            )
        )
    )
)