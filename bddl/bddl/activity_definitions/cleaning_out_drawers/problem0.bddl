(define (problem cleaning_out_drawers_0)
    (:domain igibson)

    (:objects
     	bowl.n.01_1 bowl.n.01_2 - bowl.n.01
    	cabinet.n.01_1 cabinet.n.01_2 - cabinet.n.01
    	spoon.n.01_1 spoon.n.01_2 - spoon.n.01
    	piece_of_cloth.n.01_1 - piece_of_cloth.n.01
    	sink.n.01_1 - sink.n.01
    	floor.n.01_1 - floor.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (inside bowl.n.01_1 cabinet.n.01_1) 
        (inside bowl.n.01_2 cabinet.n.01_1) 
        (inside spoon.n.01_1 cabinet.n.01_2) 
        (inside spoon.n.01_2 cabinet.n.01_2) 
        (inside piece_of_cloth.n.01_1 cabinet.n.01_1) 
        (inroom cabinet.n.01_1 kitchen) 
        (inroom cabinet.n.01_2 kitchen) 
        (inroom sink.n.01_1 kitchen) 
        (inroom floor.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (nextto ?piece_of_cloth.n.01_1 ?sink.n.01_1) 
            (nextto ?bowl.n.01_1 ?sink.n.01_1) 
            (nextto ?bowl.n.01_2 ?sink.n.01_1) 
            (nextto ?spoon.n.01_1 ?sink.n.01_1) 
            (nextto ?spoon.n.01_2 ?sink.n.01_1)
        )
    )
)