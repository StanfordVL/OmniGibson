(define (problem throwing_darts_0)
    (:domain igibson)

    (:objects
     	dartboard1 - dartboard
    	door1 - door
    	dart1 dart10 dart11 dart12 dart2 dart3 dart4 dart5 dart6 dart7 dart8 dart9 - dart
    	shelf1 - shelf
    	pool_table1 - pool_table
    )
    
    (:init 
        (nextto dartboard1 door1) 
        (ontop dart1 shelf1) 
        (ontop dart2 shelf1) 
        (ontop dart3 shelf1) 
        (ontop dart4 shelf1) 
        (ontop dart5 shelf1) 
        (ontop dart6 shelf1) 
        (ontop dart7 shelf1) 
        (ontop dart8 pool_table1) 
        (ontop dart9 pool_table1) 
        (ontop dart10 pool_table1) 
        (ontop dart11 pool_table1) 
        (ontop dart12 pool_table1) 
        (inroom door1 garage) 
        (inroom shelf1 garage) 
        (inroom pool_table1 garage)
    )
    
    (:goal 
        (and 
            (forall 
                (?dart - dart) 
                (inside ?dart ?dartboard)
            )
        )
    )
)