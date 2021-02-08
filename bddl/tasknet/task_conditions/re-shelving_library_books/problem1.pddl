(define (problem re-shelving_library_books_1)
    (:domain igibson)

    (:objects
        book1 book10 book11 book12 book2 book3 book4 book5 book6 book7 book8 book9 - book
        shelf1 shelf2 - shelf
        coffee_table1 - coffee_table
        sofa1 - sofa
    )
    
    (:init 
        (ontop book1 shelf1) 
        (ontop book2 shelf1) 
        (ontop book3 shelf2) 
        (ontop book4 shelf1) 
        (ontop book5 shelf2) 
        (ontop book6 coffee_table1) 
        (ontop book7 coffee_table1) 
        (ontop book8 coffee_table1) 
        (ontop book9 ?coffee_table) 
        (nextto book10 coffee_table1) 
        (nextto book11 coffee_table1) 
        (nextto book12 coffee_table1) 
        (inroom sofa1 living room) 
        (inroom coffee_table1 living room) 
        (inroom shelf1 living room) 
        (inroom shelf2 living room)
    )
    
    (:goal 
        (and 
            (forall 
                (?book - book) 
                (exists 
                    (?shelf - shelf) 
                    (ontop ?book ?shelf)
                )
            )
        )
    )
)