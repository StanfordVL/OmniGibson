(define (problem shooting_pool_1)
    (:domain igibson)

    (:objects
        covering1 - covering
        pool_table1 - pool_table
        carpet1 - carpet
        ball1 ball10 ball11 ball12 ball13 ball14 ball15 ball2 ball3 ball4 ball5 ball6 ball7 ball8 ball9 - ball
        stick1 stick2 - stick
    )
    
    (:init 
        (under covering1 pool_table1) 
        (ontop pool_table1 carpet1) 
        (and 
            (ontop ball1 pool_table1) 
            (nextto ball2 ball1) 
            (ontop ball2 pool_table1) 
            (nextto ball3 ball2) 
            (ontop ball3 pool_table1) 
            (nextto ball4 ball3) 
            (ontop ball4 pool_table1)
        ) 
        (and 
            (ontop stick1 pool_table1) 
            (nextto ball5 stick1) 
            (ontop ball5 pool_table1) 
            (nextto ball6 ball5) 
            (ontop ball6 pool_table1) 
            (ontop ball7 pool_table1) 
            (nextto ball7 ball6) 
            (nextto ball8 ball7) 
            (ontop ball8 pool_table1)
        ) 
        (and 
            (ontop stick2 pool_table1) 
            (ontop ball9 pool_table1) 
            (nextto ball10 ball11) 
            (ontop ball10 pool_table1) 
            (nextto ball11 stick2) 
            (ontop ball11 pool_table1) 
            (nextto ball12 stick2) 
            (ontop ball12 pool_table1) 
            (nextto ball13 ball12) 
            (ontop ball13 pool_table1) 
            (nextto ball14 ball13) 
            (ontop ball14 pool_table1) 
            (nextto ball15 ball14) 
            (ontop ball15 pool_table1)
        ) 
        (inroom pool_table1 living room) 
        (inroom carpet1 living room)
    )
    
    (:goal 
        (and 
            (forall 
                (?ball - ball) 
                (ontop ?ball ?pool_table)
            ) 
            (forall 
                (?stick - stick) 
                (nextto ?stick ?ball)
            ) 
            (exists 
                (?covering - covering) 
                (and 
                    (forall 
                        (?ball - ball) 
                        (under ?ball ?covering)
                    ) 
                    (forall 
                        (?stick - stick) 
                        (under ?stick ?covering)
                    ) 
                    (under ?pool_table ?covering)
                )
            )
        )
    )
)