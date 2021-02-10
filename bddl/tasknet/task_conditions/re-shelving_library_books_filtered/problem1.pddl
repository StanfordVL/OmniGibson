(define (problem re-shelving_library_notebooks_1)
    (:domain igibson)

    (:objects
        notebook1 notebook10 notebook11 notebook12 notebook2 notebook3 notebook4 notebook5 notebook6 notebook7 notebook8 notebook9 - notebook
        shelf1 shelf2 - shelf
        coffee_table1 - coffee_table
        sofa1 - sofa
    )
    
    (:init 
        (ontop notebook1 shelf1) 
        (ontop notebook2 shelf1) 
        (ontop notebook3 shelf2) 
        (ontop notebook4 shelf1) 
        (ontop notebook5 shelf2) 
        (ontop notebook6 coffee_table1) 
        (ontop notebook7 coffee_table1) 
        (ontop notebook8 coffee_table1) 
        (ontop notebook9 ?coffee_table) 
        (nextto notebook10 coffee_table1) 
        (nextto notebook11 coffee_table1) 
        (nextto notebook12 coffee_table1) 
        (inroom sofa1 living_room) 
        (inroom coffee_table1 living_room) 
        (inroom shelf1 living_room) 
        (inroom shelf2 living_room)
    )
    
    (:goal 
        (and 
            (forall 
                (?notebook - notebook) 
                (exists 
                    (?shelf - shelf) 
                    (ontop ?notebook ?shelf)
                )
            )
        )
    )
)
