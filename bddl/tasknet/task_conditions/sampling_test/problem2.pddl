
(define (problem packlunch)
    (:domain igibson)
    (:objects
        chip1 chip2 chip3 chip4 - chip        
        table1 - table
    )
    (:init
        (under chip1 table1)
        (under chip2 table1)
        (under chip3 table1)
        (under chip4 table1)
    )
    (:goal
        (and 
        (not (under chip1 table1))
        (not (under chip2 table1))
        (not (under chip3 table1))
        (not (under chip4 table1))
        )
    )
)