(define (problem putting_dishes_away_after_cleaning_0)
    (:domain igibson)

    (:objects
        dishwasher.n.01_1 - dishwasher.n.01
        cabinet.n.01_1 - cabinet.n.01
        dish.n.01_1 - dish.n.01
    )
    
    (:init 
        (inside dish.n.01_1 dishwasher.n.01_1) 
        (inroom cabinet.n.01_1 kitchen)
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
        )
    )
)
