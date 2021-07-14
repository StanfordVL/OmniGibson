(define (problem cleaning_up_after_a_meal_1)
    (:domain igibson)

    (:objects
     	plate.n.04_1 plate.n.04_2 plate.n.04_3 plate.n.04_4 - plate.n.04
    	table.n.02_1 - table.n.02
    	glass.n.02_1 glass.n.02_2 glass.n.02_3 glass.n.02_4 - glass.n.02
    	sink.n.01_1 - sink.n.01
        floor.n.01_1 - floor.n.01
        agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (ontop plate.n.04_1 table.n.02_1) 
        (ontop plate.n.04_2 table.n.02_1) 
        (ontop plate.n.04_3 table.n.02_1) 
        (ontop plate.n.04_4 table.n.02_1) 
        (ontop glass.n.02_1 table.n.02_1) 
        (ontop glass.n.02_2 table.n.02_1) 
        (ontop glass.n.02_3 table.n.02_1) 
        (ontop glass.n.02_4 table.n.02_1) 
        (inroom sink.n.01_1 kitchen) 
        (inroom table.n.02_1 dining_room) 
        (inroom floor.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forall 
                (?plate.n.04 - plate.n.04) 
                (nextto ?plate.n.04 ?sink.n.01_1)
            ) 
            (forall 
                (?glass.n.02 - glass.n.02) 
                (nextto ?glass.n.02 ?sink.n.01_1)
            )
        )
    )
)