(define (problem packlunch)
    (:domain igibson)
    (:objects
        chip1 chip2 chip3 chip4 chip5 chip6 - chip        
        counter1 - counter
        fridge1 - fridge
        table1 - table
    )
    (:init
        (inRoom counter1 kitchen)
        (inRoom fridge1 kitchen)
        (onTop chip1 counter1)
        (onTop chip2 counter1)
        (inside chip3 fridge1)
        (inside chip4 fridge1)
        (under chip5 table1)
        (under chip6 table1)
    )
    (:goal
        (and 
        (not (onTop chip1 counter1))
        (not (onTop chip2 counter1))
        (not (inside chip3 fridge1))
        (not (inside chip4 fridge1))
        (not (under chip5 table1))
        (not (under chip6 table1))
        )
    )
)