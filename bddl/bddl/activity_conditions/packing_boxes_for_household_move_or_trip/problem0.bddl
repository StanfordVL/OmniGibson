(define (problem packing_boxes_for_household_move_or_trip_0)
    (:domain igibson)

    (:objects
     	carton.n.02_1 carton.n.02_2 - carton.n.02
    	floor.n.01_1 - floor.n.01
    	book.n.02_1 book.n.02_2 - book.n.02
    	table.n.02_1 - table.n.02
    	sweater.n.01_1 - sweater.n.01
    	shirt.n.01_1 shirt.n.01_2 - shirt.n.01
    	plate.n.04_1 plate.n.04_2 plate.n.04_3 plate.n.04_4 - plate.n.04
    	dishtowel.n.01_1 - dishtowel.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor carton.n.02_1 floor.n.01_1) 
        (onfloor carton.n.02_2 floor.n.01_1) 
        (ontop book.n.02_2 table.n.02_1) 
        (ontop book.n.02_1 table.n.02_1) 
        (onfloor sweater.n.01_1 floor.n.01_1) 
        (onfloor shirt.n.01_1 floor.n.01_1) 
        (onfloor shirt.n.01_2 floor.n.01_1) 
        (ontop plate.n.04_1 table.n.02_1) 
        (onfloor plate.n.04_2 floor.n.01_1) 
        (onfloor plate.n.04_3 floor.n.01_1) 
        (onfloor plate.n.04_4 floor.n.01_1) 
        (onfloor dishtowel.n.01_1 floor.n.01_1) 
        (inroom floor.n.01_1 living_room) 
        (inroom table.n.02_1 living_room) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (exists 
                (?carton.n.02 - carton.n.02) 
                (and 
                    (forall 
                        (?plate.n.04 - plate.n.04) 
                        (inside ?plate.n.04 ?carton.n.02)
                    ) 
                    (inside ?dishtowel.n.01_1 ?carton.n.02)
                )
            ) 
            (exists 
                (?carton.n.02 - carton.n.02) 
                (and 
                    (forall 
                        (?book.n.02 - book.n.02) 
                        (inside ?book.n.02 ?carton.n.02)
                    ) 
                    (forall 
                        (?shirt.n.01 - shirt.n.01) 
                        (inside ?shirt.n.01 ?carton.n.02)
                    ) 
                    (inside ?sweater.n.01_1 ?carton.n.02)
                )
            )
        )
    )
)