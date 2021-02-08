(define (problem re-shelving_library_notebooks_0)
    (:domain igibson)

    (:objects
        notebook1 notebook10 notebook11 notebook12 notebook2 notebook3 notebook4 notebook5 notebook6 notebook7 notebook8 notebook9 - notebook
        table1 - table
    )
    
    (:init 
        (ontop notebook1 table1) 
        (ontop notebook2 table1) 
        (ontop notebook3 table1) 
        (ontop notebook4 table1) 
        (ontop notebook5 table1) 
        (ontop notebook6 table1) 
        (ontop notebook7 table1) 
        (ontop notebook8 table1) 
        (ontop notebook9 table1) 
        (ontop notebook10 table1) 
        (ontop notebook11 table1) 
        (ontop notebook12 table1)
    )
    
    (:goal 
        (and 
            (and 
                (forall 
                    (?notebook - notebook) 
                    (inside ?notebook ?shelf1)
                ) 
                (forall 
                    (?notebook - notebook) 
                    (not 
                        (ontop ?notebook ?table1)
                    )
                )
            )
        )
    )
)
