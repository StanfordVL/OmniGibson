(define (problem sorting_books_0)
    (:domain igibson)

    (:objects
     	coffee_table1 - coffee_table
    	floor1 - floor
    	sofa1 - sofa
    	sofa_chair1 - sofa_chair
    	shelf1 shelf2 - shelf
    	wall1 - wall
    	hardback1 hardback2 hardback3 hardback4 hardback5 hardback6 hardback7 hardback8 hardback9 hardback10 - hardback
    	notebook1 notebook2 notebook3 notebook4 notebook5 notebook6 notebook7 notebook8 notebook9 notebook10 notebook11 notebook12 notebook13 notebook14 - notebook
    )
    
    (:init 
        (ontop coffee_table1 floor1) 
        (ontop sofa1 floor1) 
        (ontop sofa_chair1 floor1) 
        (nextto shelf1 wall1) 
        (nextto shelf2 wall1) 
        (ontop hardback9 shelf1) 
        (ontop hardback10 shelf1) 
        (ontop notebook9 coffee_table1) 
        (ontop notebook10 sofa_chair1) 
        (ontop notebook11 floor1) 
        (ontop notebook12 shelf1) 
        (ontop notebook13 shelf2) 
        (ontop notebook14 coffee_table1) 
        (ontop hardback1 shelf1) 
        (ontop hardback2 shelf2) 
        (ontop hardback3 coffee_table1) 
        (ontop hardback4 sofa1) 
        (ontop hardback5 coffee_table1) 
        (ontop hardback6 sofa_chair1) 
        (ontop hardback7 floor1) 
        (ontop hardback8 floor1) 
        (ontop notebook1 coffee_table1) 
        (ontop notebook2 shelf1) 
        (ontop notebook3 coffee_table1) 
        (ontop notebook4 floor1) 
        (ontop notebook5 sofa_chair1) 
        (ontop notebook6 shelf2) 
        (ontop notebook7 shelf1) 
        (ontop notebook8 coffee_table1) 
        (inroom shelf1 living_room) 
        (inroom shelf2 living_room) 
        (inroom coffee_table1 living_room) 
        (inroom sofa1 living_room) 
        (inroom sofa_chair1 living_room) 
        (inroom floor1 living_room) 
        (inroom wall1 living_room)
    )
    
    (:goal 
        (and 
            (forall 
                (?hardback - hardback) 
                (ontop ?hardback ?shelf1)
            ) 
            (forall 
                (?notebook - notebook) 
                (ontop ?notebook ?shelf2)
            )
        )
    )
)
