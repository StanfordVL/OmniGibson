(define (problem throwing_darts_1)
    (:domain igibson)

    (:objects
     	coffee_table1 - coffee_table
    	wall1 - wall
    	dart1 dart10 dart11 dart12 dart13 dart14 dart15 dart16 dart2 dart3 dart4 dart5 dart6 dart7 dart8 dart9 - dart
    	dartboard1 - dartboard
    )
    
    (:init 
        (nextto coffee_table1 wall1) 
        (ontop dart1 coffee_table1) 
        (ontop dart2 coffee_table1) 
        (ontop dart3 coffee_table1) 
        (ontop dart4 coffee_table1) 
        (ontop dart5 coffee_table1) 
        (ontop dart6 coffee_table1) 
        (ontop dart7 coffee_table1) 
        (ontop dart8 coffee_table1) 
        (ontop dart9 coffee_table1) 
        (ontop dart10 coffee_table1) 
        (ontop dart11 coffee_table1) 
        (ontop dart12 coffee_table1) 
        (ontop dart13 coffee_table1) 
        (ontop dart14 coffee_table1) 
        (ontop dart15 coffee_table1) 
        (ontop dart16 coffee_table1) 
        (nextto dartboard1 wall1) 
        (inroom wall1 living_room) 
        (inroom coffee_table1 living_room)
    )
    
    (:goal 
        (and 
            (nextto ?coffee_table1 ?wall1) 
            (nextto ?dartboard1 ?wall1) 
            (forall 
                (?dart - dart) 
                (ontop ?dart ?dartboard1)
            )
        )
    )
)
