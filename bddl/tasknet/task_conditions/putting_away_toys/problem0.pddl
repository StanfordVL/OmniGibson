(define (problem putting_away_toys_0)
    (:domain igibson)

    (:objects
     	plaything.n.01_1 plaything.n.01_2 plaything.n.01_3 plaything.n.01_4 plaything.n.01_5 plaything.n.01_6 plaything.n.01_7 plaything.n.01_8 - plaything.n.01
    	floor.n.01_1 - floor.n.01
    	container.n.01_1 - container.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor plaything.n.01_1 floor.n.01_1) 
        (onfloor plaything.n.01_2 floor.n.01_1) 
        (onfloor plaything.n.01_3 floor.n.01_1) 
        (onfloor plaything.n.01_4 floor.n.01_1) 
        (onfloor plaything.n.01_5 floor.n.01_1) 
        (onfloor plaything.n.01_6 floor.n.01_1) 
        (onfloor plaything.n.01_7 floor.n.01_1) 
        (onfloor plaything.n.01_8 floor.n.01_1) 
        (onfloor container.n.01_1 floor.n.01_1) 
        (inroom floor.n.01_1 living_room) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forall 
                (?plaything.n.01 - plaything.n.01) 
                (inside ?plaything.n.01 ?container.n.01_1)
            )
        )
    )
)