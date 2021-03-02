(define (problem shooting_pool_0)
    (:domain igibson)

    (:objects
        grandfather_clock1 - grandfather_clock
        pool_table1 - pool_table
        floor_lamp1 - floor_lamp
        sofa1 - sofa
        stick1 - stick
        ball1 ball10 ball11 ball12 ball13 ball14 ball15 ball16 ball17 ball18 ball19 ball2 ball20 ball21 ball3 ball4 ball5 ball6 ball7 ball8 ball9 - ball
    )
    
    (:init 
        (imply 
            (nextto grandfather_clock1 pool_table1) 
            (nextto floor_lamp1 grandfather_clock1)
        ) 
        (and 
            (nextto sofa1 floor_lamp1) 
            (ontop stick1 pool_table1) 
            (ontop ball1 pool_table1) 
            (ontop ball2 pool_table1) 
            (ontop ball3 pool_table1) 
            (ontop ball4 pool_table1) 
            (ontop ball5 pool_table1) 
            (ontop ball6 pool_table1) 
            (ontop ball7 pool_table1) 
            (ontop ball8 pool_table1) 
            (ontop ball9 pool_table1) 
            (ontop ball10 pool_table1) 
            (ontop ball11 pool_table1) 
            (ontop ball12 pool_table1) 
            (ontop ball13 pool_table1) 
            (ontop ball14 pool_table1) 
            (ontop ball15 pool_table1) 
            (ontop ball16 pool_table1) 
            (ontop ball17 pool_table1) 
            (ontop ball18 pool_table1) 
            (ontop ball19 pool_table1) 
            (ontop ball20 pool_table1) 
            (ontop ball21 pool_table1) 
            (nextto pool_table1 floor_lamp1)
        )
    )
    
    (:goal 
        (and 
            (forn 
                (21) 
                (?ball - ball) 
                (inside ?ball ?pool_table1)
            ) 
            (nextto ?sofa1 ?floor_lamp1) 
            (nextto ?floor_lamp1 ?grandfather_clock1) 
            (ontop ?stick1 ?pool_table1) 
            (nextto ?grandfather_clock1 ?pool_table1)
        )
    )
)