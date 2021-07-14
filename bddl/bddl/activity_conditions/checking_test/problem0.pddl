(define (problem checking_test)
    (:domain igibson)
    (:objects
        chicken1 chicken2 chicken3 chicken4 - chicken
        apple1 apple2 apple3 - apple
        
    )
    (:init
        (nextto chicken1 chicken2)
        (not 
            (cooked chicken1)
        )
        (not 
            (cooked chicken2)
        )
        (not 
            (cooked chicken3)
        )
        (not 
            (cooked chicken4)
        )
        (not 
            (cooked apple1)
        )
        (not 
            (cooked apple2)
        )
        (not 
            (cooked apple3)
        )
        (or 
            (nextto apple1 chicken1)
            (nextto apple1 chicken4)
        )
    )
    (:goal
        (and 
            (forall 
                (?chicken - chicken) 
                (cooked ?chicken)
            )
            (forn 
                (2) 
                (?chicken - chicken) 
                (cooked ?chicken)
            )
            (exists 
                (?chicken - chicken) 
                (cooked ?chicken)
            )
            (exists 
                (?apple - apple) 
                (not 
                    (cooked ?apple)
                )
            )
            (or
                (cooked apple1)
                (cooked apple2)
            )
            (imply
                (cooked apple1)
                (cooked apple2)
            )
        )
    )
)