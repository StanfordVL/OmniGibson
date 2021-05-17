(define (problem cleaning_high_chair_0)
    (:domain igibson)

    (:objects
        highchair.n.01_1 - highchair.n.01
        crumb.n.03_1 crumb.n.03_2 crumb.n.03_3 crumb.n.03_4 - crumb.n.03
        piece_of_cloth.n.01_1 - piece_of_cloth.n.01
        cabinet.n.01_1 - cabinet.n.01
        sink.n.01_1 - sink.n.01
        floor.n.01_1 floor.n.01_2 - floor.n.01
        agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (dusty highchair.n.01_1) 
        (ontop crumb.n.03_1 highchair.n.01_1) 
        (ontop crumb.n.03_2 highchair.n.01_1) 
        (ontop crumb.n.03_3 highchair.n.01_1) 
        (ontop crumb.n.03_4 highchair.n.01_1) 
        (inside piece_of_cloth.n.01_1 cabinet.n.01_1) 
        (onfloor highchair.n.01_1 floor.n.01_2) 
        (inroom cabinet.n.01_1 kitchen) 
        (inroom sink.n.01_1 kitchen) 
        (inroom floor.n.01_2 dining_room) 
        (inroom floor.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forall 
                (?crumb.n.03 - crumb.n.03) 
                (inside ?crumb.n.03 ?sink.n.01_1)
            ) 
            (not 
                (dusty ?highchair.n.01_1)
            )
        )
    )
)