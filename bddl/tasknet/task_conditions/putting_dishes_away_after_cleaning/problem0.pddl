(define (problem putting_dishes_away_after_cleaning_0)
    (:domain igibson)

    (:objects
        dishwasher1 - dishwasher
        dish1 dish10 dish11 dish12 dish2 dish3 dish4 dish5 dish6 dish7 dish8 dish9 - dish
    )
    
    (:init 
        (open dishwasher1) 
        (and 
            (inside dish1 dishwasher1) 
            (scrubbed dish1)
        ) 
        (and 
            (inside dish2 dishwasher1) 
            (scrubbed dish2)
        ) 
        (and 
            (inside dish3 dishwasher1) 
            (scrubbed dish3)
        ) 
        (and 
            (inside dish4 dishwasher1) 
            (scrubbed dish4)
        ) 
        (and 
            (inside dish5 dishwasher1) 
            (scrubbed dish5)
        ) 
        (and 
            (inside dish6 dishwasher1) 
            (scrubbed dish6)
        ) 
        (and 
            (inside dish7 dishwasher1) 
            (scrubbed dish7)
        ) 
        (and 
            (inside dish8 dishwasher1) 
            (scrubbed dish8)
        ) 
        (and 
            (inside dish9 dishwasher1) 
            (scrubbed dish9)
        ) 
        (and 
            (inside dish10 dishwasher1) 
            (scrubbed dish10)
        ) 
        (and 
            (inside dish11 dishwasher1) 
            (scrubbed dish11)
        ) 
        (and 
            (inside dish12 dishwasher1) 
            (scrubbed dish12)
        )
    )
    
    (:goal 
        (and 
            (exists 
                (?cabinet - cabinet) 
                (forall 
                    (?dish - dish) 
                    (inside ?dish ?cabinet)
                )
            ) 
            (not 
                (open ?dishwasher)
            )
        )
    )
)