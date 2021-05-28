(define (problem cleaning_up_after_a_meal_0)
    (:domain igibson)

    (:objects
     	bowl.n.01_1 bowl.n.01_2 - bowl.n.01
    	table.n.02_1 - table.n.02
    	sack.n.01_1 - sack.n.01
    	chair.n.01_1 chair.n.01_2 - chair.n.01
    	plate.n.04_1 plate.n.04_2 plate.n.04_3 plate.n.04_4 - plate.n.04
    	cup.n.01_1 cup.n.01_2 - cup.n.01
    	hamburger.n.01_1 hamburger.n.01_2 - hamburger.n.01
    	floor.n.01_1 floor.n.01_2 - floor.n.01
    	detergent.n.02_1 - detergent.n.02
    	dishwasher.n.01_1 - dishwasher.n.01
        sink.n.01_1 - sink.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (ontop bowl.n.01_1 table.n.02_1) 
        (ontop bowl.n.01_2 table.n.02_1) 
        (stained bowl.n.01_1) 
        (stained bowl.n.01_2) 
        (ontop sack.n.01_1 table.n.02_1) 
        (ontop plate.n.04_1 table.n.02_1) 
        (ontop plate.n.04_2 table.n.02_1) 
        (ontop plate.n.04_3 table.n.02_1) 
        (ontop plate.n.04_4 table.n.02_1) 
        (stained plate.n.04_1) 
        (stained plate.n.04_2) 
        (stained plate.n.04_3) 
        (stained plate.n.04_4) 
        (ontop cup.n.01_1 table.n.02_1) 
        (ontop cup.n.01_2 table.n.02_1) 
        (stained cup.n.01_1) 
        (stained cup.n.01_2) 
        (ontop hamburger.n.01_1 chair.n.01_2) 
        (onfloor hamburger.n.01_2 floor.n.01_1) 
        (onfloor detergent.n.02_1 floor.n.01_1) 
        (stained chair.n.01_1) 
        (stained chair.n.01_2) 
        (stained table.n.02_1) 
        (inroom dishwasher.n.01_1 kitchen) 
        (inroom floor.n.01_1 dining_room) 
        (inroom floor.n.01_2 kitchen) 
        (inroom table.n.02_1 dining_room) 
        (inroom chair.n.01_1 dining_room) 
        (inroom chair.n.01_2 dining_room) 
        (inroom sink.n.01_1 kitchen)
        (onfloor agent.n.01_1 floor.n.01_2)
    )
    
    (:goal 
        (and 
            (forall 
                (?bowl.n.01 - bowl.n.01) 
                (not 
                    (stained ?bowl.n.01)
                )
            ) 
            (forall 
                (?plate.n.04 - plate.n.04) 
                (not 
                    (stained ?plate.n.04)
                )
            ) 
            (forall 
                (?cup.n.01 - cup.n.01) 
                (not 
                    (stained ?cup.n.01)
                )
            ) 
            (forall 
                (?hamburger.n.01 - hamburger.n.01) 
                (inside ?hamburger.n.01 ?sack.n.01_1)
            ) 
            (onfloor ?sack.n.01_1 ?floor.n.01_1) 
            (not 
                (stained ?floor.n.01_1)
            ) 
            (not 
                (stained ?chair.n.01_2)
            ) 
            (not 
                (stained ?floor.n.01_1)
            ) 
            (not 
                (stained ?table.n.02_1)
            )
        )
    )
)