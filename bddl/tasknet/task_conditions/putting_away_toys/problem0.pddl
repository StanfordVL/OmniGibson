(define (problem putting_away_toys_0)
    (:domain igibson)

    (:objects
        plaything.n.01_1 plaything.n.01_2 plaything.n.01_3 plaything.n.01_4 plaything.n.01_5 plaything.n.01_6 plaything.n.01_7 plaything.n.01_8 - plaything.n.01
        floor.n.01_1 floor.n.01_2 - floor.n.01
        carton.n.02_1 carton.n.02_2 - carton.n.02
        agent.n.01_1 - agent.n.01
        table.n.02_1 - table.n.02
    )
    
    (:init 
        (onfloor plaything.n.01_1 floor.n.01_1) 
        (onfloor plaything.n.01_2 floor.n.01_1) 
        (onfloor plaything.n.01_3 floor.n.01_1) 
        (onfloor plaything.n.01_4 floor.n.01_1) 
        (onfloor plaything.n.01_5 floor.n.01_2) 
        (onfloor plaything.n.01_6 floor.n.01_2) 
        (onfloor plaything.n.01_7 floor.n.01_2) 
        (onfloor plaything.n.01_8 floor.n.01_2) 
        (onfloor carton.n.02_1 floor.n.01_1) 
        (ontop carton.n.02_2 table.n.02_1) 
        (inroom floor.n.01_1 living_room) 
        (inroom floor.n.01_2 dining_room) 
        (inroom table.n.02_1 dining_room) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forall 
                (?plaything.n.01 - plaything.n.01) 
                (exists 
                    (?carton.n.02 - carton.n.02) 
                    (inside ?plaything.n.01 ?carton.n.02)
                )
            )
        )
    )
)