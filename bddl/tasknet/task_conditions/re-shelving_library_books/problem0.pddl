(define (problem re-shelving_library_books_0)
    (:domain igibson)

    (:objects
        book1 book10 book11 book12 book2 book3 book4 book5 book6 book7 book8 book9 - book
        table1 - table
    )
    
    (:init 
        (ontop book1 table1) 
        (ontop book2 table1) 
        (ontop book3 table1) 
        (ontop book4 table1) 
        (ontop book5 table1) 
        (ontop book6 table1) 
        (ontop book7 table1) 
        (ontop book8 table1) 
        (ontop book9 table1) 
        (ontop book10 table1) 
        (ontop book11 table1) 
        (ontop book12 table1)
    )
    
    (:goal 
        (and 
            (and 
                (forall 
                    (?book - book) 
                    (inside ?book ?shelf1)
                ) 
                (forall 
                    (?book - book) 
                    (not 
                        (ontop ?book ?table1)
                    )
                )
            )
        )
    )
)