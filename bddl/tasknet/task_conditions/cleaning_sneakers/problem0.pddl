(define (problem cleaning_sneakers_0)
    (:domain igibson)

    (:objects
     	gym_shoe1 gym_shoe2 gym_shoe3 gym_shoe4 - gym_shoe
    	floor1 - floor
    	wall1 - wall
    	cleansing_agent1 - cleansing_agent
    	bench1 - bench
    	piece_of_cloth1 - piece_of_cloth
    )
    
    (:init 
        (dusty gym_shoe1) 
        (and 
            (ontop gym_shoe1 floor1)
        ) 
        (dusty gym_shoe2) 
        (and 
            (nextto gym_shoe2 wall1)
        ) 
        (dusty gym_shoe3) 
        (and 
            (ontop gym_shoe3 floor1) 
            (nextto gym_shoe3 gym_shoe1)
        ) 
        (dusty gym_shoe4) 
        (and 
            (nextto gym_shoe4 wall1) 
            (nextto gym_shoe4 gym_shoe2)
        ) 
        (ontop cleansing_agent1 bench1) 
        (ontop piece_of_cloth1 bench1) 
        (inroom floor1 garage) 
        (inroom bench1 garage) 
        (inroom wall1 garage)
    )
    
    (:goal 
        (and 
            (forall 
                (?gym_shoe - gym_shoe) 
                (and 
                    (ontop ?gym_shoe ?bench1) 
                    (scrubbed ?gym_shoe)
                )
            ) 
            (exists 
                (?gym_shoe - gym_shoe) 
                (nextto ?gym_shoe ?gym_shoe1)
            ) 
            (exists 
                (?gym_shoe - gym_shoe) 
                (nextto ?gym_shoe ?gym_shoe2)
            ) 
            (exists 
                (?gym_shoe - gym_shoe) 
                (nextto ?gym_shoe ?gym_shoe3)
            ) 
            (exists 
                (?gym_shoe - gym_shoe) 
                (nextto ?gym_shoe ?gym_shoe4)
            ) 
            (ontop ?piece_of_cloth1 ?bench1) 
            (ontop ?cleansing_agent1 ?bench1)
        )
    )
)