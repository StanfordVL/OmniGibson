(define (problem sorting_books_0)
    (:domain igibson)

    (:objects
     	coffee_table1 - coffee_table
    	floor1 - floor
    	sofa1 - sofa
    	sofa_chair1 - sofa_chair
    	shelf1 shelf2 shelf3 - shelf
    	wall1 - wall
    	paperback_book1 paperback_book2 paperback_book3 paperback_book4 paperback_book5 paperback_book6 paperback_book7 paperback_book8 - paperback_book
    	hardback1 hardback2 hardback3 hardback4 hardback5 hardback6 hardback7 hardback8 - hardback
    	novel1 novel2 novel3 novel4 novel5 novel6 novel7 novel8 - novel
    )
    
    (:init 
        (ontop coffee_table1 floor1) 
        (ontop sofa1 floor1) 
        (ontop sofa_chair1 floor1) 
        (nextto shelf1 wall1) 
        (nextto shelf2 wall1) 
        (nextto shelf3 wall1) 
        (ontop paperback_book1 shelf1) 
        (ontop paperback_book2 shelf1) 
        (ontop paperback_book3 coffee_table1) 
        (ontop paperback_book4 sofa_chair1) 
        (ontop paperback_book5 floor1) 
        (ontop paperback_book6 shelf3) 
        (ontop paperback_book7 shelf3) 
        (ontop paperback_book8 coffee_table1) 
        (ontop hardback1 shelf3) 
        (ontop hardback2 shelf2) 
        (ontop hardback3 coffee_table1) 
        (ontop hardback4 sofa1) 
        (ontop hardback5 coffee_table1) 
        (ontop hardback6 sofa_chair1) 
        (ontop hardback7 floor1) 
        (ontop hardback8 floor1) 
        (ontop novel1 coffee_table1) 
        (ontop novel2 shelf1) 
        (ontop novel3 coffee_table1) 
        (ontop novel4 floor1) 
        (ontop novel5 sofa_chair1) 
        (ontop novel6 shelf2) 
        (ontop novel7 shelf1) 
        (ontop novel8 coffee_table1) 
        (inroom shelf1 living room) 
        (inroom shelf2 living room) 
        (inroom shelf3 living room) 
        (inroom coffee_table1 living room) 
        (inroom sofa1 living room) 
        (inroom sofa_chair1 living room) 
        (inroom floor1 living room) 
        (inroom wall1 living room)
    )
    
    (:goal 
        (and 
            (forall 
                (?hardback - hardback) 
                (ontop ?hardback ?shelf1)
            ) 
            (forall 
                (?paperback_book - paperback_book) 
                (ontop ?paperback_book ?shelf2)
            ) 
            (forall 
                (?novel - novel) 
                (ontop ?novel ?shelf3)
            )
        )
    )
)