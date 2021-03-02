(define (problem putting_dishes_away_after_cleaning_0)
    (:domain igibson)

    (:objects
        dishwasher.n.01_1 - dishwasher.n.01
        dish.n.01_1 dish.n.01_10 dish.n.01_11 dish.n.01_12 dish.n.01_2 dish.n.01_3 dish.n.01_4 dish.n.01_5 dish.n.01_6 dish.n.01_7 dish.n.01_8 dish.n.01_9 - dish.n.01
    )
    
    (:init 
        (open dishwasher.n.01_1) 
        (and 
            (inside dish.n.01_1 dishwasher.n.01_1) 
            (scrubbed dish.n.01_1)
        ) 
        (and 
            (inside dish.n.01_2 dishwasher.n.01_1) 
            (scrubbed dish.n.01_2)
        ) 
        (and 
            (inside dish.n.01_3 dishwasher.n.01_1) 
            (scrubbed dish.n.01_3)
        ) 
        (and 
            (inside dish.n.01_4 dishwasher.n.01_1) 
            (scrubbed dish.n.01_4)
        ) 
        (and 
            (inside dish.n.01_5 dishwasher.n.01_1) 
            (scrubbed dish.n.01_5)
        ) 
        (and 
            (inside dish.n.01_6 dishwasher.n.01_1) 
            (scrubbed dish.n.01_6)
        ) 
        (and 
            (inside dish.n.01_7 dishwasher.n.01_1) 
            (scrubbed dish.n.01_7)
        ) 
        (and 
            (inside dish.n.01_8 dishwasher.n.01_1) 
            (scrubbed dish.n.01_8)
        ) 
        (and 
            (inside dish.n.01_9 dishwasher.n.01_1) 
            (scrubbed dish.n.01_9)
        ) 
        (and 
            (inside dish.n.01_10 dishwasher.n.01_1) 
            (scrubbed dish.n.01_10)
        ) 
        (and 
            (inside dish.n.01_11 dishwasher.n.01_1) 
            (scrubbed dish.n.01_11)
        ) 
        (and 
            (inside dish.n.01_12 dishwasher.n.01_1) 
            (scrubbed dish.n.01_12)
        )
        (inroom dishwasher.n.01_1 kitchen)
    )
    
    (:goal 
        (and 
            (exists 
                (?cabinet.n.01 - cabinet.n.01) 
                (forall 
                    (?dish.n.01 - dish.n.01) 
                    (inside ?dish.n.01 ?cabinet.n.01)
                )
            ) 
            (not 
                (open ?dishwasher.n.01)
            )
        )
    )
)
