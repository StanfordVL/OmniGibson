(define (problem packlunch)
    (:domain igibson)
    (:objects
        chip1 chip2 chip3 chip4 - chip        
        fridge1 - fridge
    )
    (:init
        (inside chip1 fridge1)
        (inside chip2 fridge1)
        (inside chip3 fridge1)
        (inside chip4 fridge1)
    )
    (:goal
        (and 
        (inside (under chip1 fridge1))
        (inside (under chip2 fridge1))
        (inside (under chip3 fridge1))
        (inside (under chip4 fridge1))
        )
    )
)