(define (problem cleaning_sneakers_1)
    (:domain igibson)

    (:objects
     	gym_shoe1 gym_shoe2 gym_shoe3 gym_shoe4 - gym_shoe
    	bed1 - bed
    	chest1 - chest
    	soap1 - soap
    	cabinet1 - cabinet
    	washcloth1 - washcloth
    	toothbrush1 - toothbrush
    	counter1 - counter
    )
    
    (:init 
        (and 
            (not 
                (scrubbed gym_shoe1)
            ) 
            (not 
                (scrubbed gym_shoe2)
            ) 
            (not 
                (scrubbed gym_shoe3)
            ) 
            (not 
                (scrubbed gym_shoe4)
            )
        ) 
        (imply 
            (under gym_shoe1 bed1) 
            (under gym_shoe2 bed1)
        ) 
        (imply 
            (inside gym_shoe3 chest1) 
            (inside gym_shoe4 chest1)
        ) 
        (and 
            (inside soap1 cabinet1) 
            (inside washcloth1 cabinet1) 
            (inside toothbrush1 cabinet1)
        ) 
        (inroom cabinet1 bathroom) 
        (inroom bed1 bathroom) 
        (inroom counter1 bathroom) 
        (inroom chest1 bathroom)
    )
    
    (:goal 
        (and 
            (forall 
                (?gym_shoe - gym_shoe) 
                (scrubbed ?gym_shoe)
            ) 
            (and 
                (not 
                    (under ?gym_shoe1 ?bed1)
                ) 
                (not 
                    (under ?gym_shoe2 ?bed1)
                )
            ) 
            (and 
                (not 
                    (inside ?gym_shoe3 ?chest1)
                ) 
                (not 
                    (under ?gym_shoe4 ?chest1)
                )
            ) 
            (and 
                (ontop ?soap1 ?counter1) 
                (ontop ?washcloth1 ?counter1) 
                (ontop ?toothbrush1 ?counter1)
            )
        )
    )
)