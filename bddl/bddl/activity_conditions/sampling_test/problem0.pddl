(define (problem packlunch)
    (:domain igibson)
    (:objects
        chip1 chip2 chip3 chip4 - chip        
        counter1 - counter
    )
    (:init
        (onTop chip1 counter1)
        (onTop chip2 counter1)
        (onTop chip3 counter1)
        (onTop chip4 counter1)
    )
    (:goal
        (and 
        (not (onTop chip1 counter1))
        (not (onTop chip2 counter1))
        (not (onTop chip3 counter1))
        (not (onTop chip4 counter1))
        )
    )
)