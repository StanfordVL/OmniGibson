(define (problem putting_dishes_away_after_cleaning_0)
    (:domain igibson)

    (:objects
        countertop.n.01_1 - countertop.n.01
        cabinet.n.01_1 - cabinet.n.01
        plate.n.04_1 plate.n.04_2 - plate.n.04
    )
    
    (:init 
        (ontop plate.n.04_1 countertop.n.01_1)
        (ontop plate.n.04_2 countertop.n.01_1)
        (inroom cabinet.n.01_1 kitchen)
        (inroom countertop.n.01_1 kitchen)
    )
    
    (:goal 
        (and 
            (exists 
                (?cabinet.n.01 - cabinet.n.01) 
                (forall 
                    (?plate.n.04 - plate.n.04)
                    (inside ?plate.n.04 ?cabinet.n.01)
                )
            ) 
        )
    )
)
