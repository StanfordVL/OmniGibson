(define (problem sorting_books_1)
    (:domain igibson)

    (:objects
     	novel1 novel2 novel3 novel4 novel5 - novel
    	table1 - table
    	shelf1 - shelf
    	chair1 - chair
    	hardback1 hardback2 hardback3 hardback4 - hardback
    	paperback_book1 paperback_book2 paperback_book3 - paperback_book
    )
    
    (:init 
        (ontop novel1 table1) 
        (ontop novel2 shelf1) 
        (ontop novel3 chair1) 
        (ontop novel4 table1) 
        (ontop novel5 chair1) 
        (ontop hardback1 table1) 
        (ontop hardback2 shelf1) 
        (ontop hardback3 shelf1) 
        (ontop hardback4 shelf1) 
        (ontop paperback_book1 chair1) 
        (ontop paperback_book2 table1) 
        (ontop paperback_book3 shelf1) 
        (inroom table1 living room) 
        (inroom shelf1 living room) 
        (inroom chair1 living room)
    )
    
    (:goal 
        (and 
            (forall 
                (?novel - novel) 
                (or 
                    (nextto ?novel ?novel1) 
                    (nextto ?novel ?novel2) 
                    (nextto ?novel ?novel3) 
                    (nextto ?novel ?novel4) 
                    (nextto ?novel ?novel5)
                )
            ) 
            (forall 
                (?hardback - hardback) 
                (or 
                    (nextto ?hardback ?hardback1) 
                    (nextto ?hardback ?hardback2) 
                    (nextto ?hardback ?hardback3) 
                    (nextto ?hardback ?hardback4)
                )
            ) 
            (forall 
                (?paperback_book - paperback_book) 
                (or 
                    (nextto ?paperback_book ?paperback_book1) 
                    (nextto ?paperback_book ?paperback_book2) 
                    (nextto ?paperback_book ?paperback_book3)
                )
            ) 
            (scrubbed ?table1)
        )
    )
)