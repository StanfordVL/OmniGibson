(define (problem decorating_the_Christmas_tree_0)
    (:domain igibson)

    (:objects
     	tree1 - tree
    	stand1 - stand
    	window1 - window
    	box1 - box
    	carpet1 - carpet
    	light1 light2 light3 - light
    	bell1 bell2 bell3 bell4 - bell
    	tinsel1 tinsel2 - tinsel
    	wreath1 - wreath
    	topper1 - topper
    	skirt1 - skirt
    )
    
    (:init 
        (and 
            (inside tree1 stand1) 
            (nextto tree1 window1)
        ) 
        (and 
            (ontop box1 carpet1) 
            (open box1)
        ) 
        (and 
            (and 
                (inside light1 box1) 
                (under carpet1 light1)
            ) 
            (and 
                (inside light2 box1) 
                (under carpet1 light2)
            ) 
            (and 
                (inside light3 box1) 
                (under carpet1 light3)
            )
        ) 
        (and 
            (and 
                (inside bell1 box1) 
                (under carpet1 bell1)
            ) 
            (and 
                (inside bell2 box1) 
                (under carpet1 bell2)
            ) 
            (and 
                (inside bell3 box1) 
                (under carpet1 bell3)
            ) 
            (and 
                (inside bell4 box1) 
                (under carpet1 bell4)
            )
        ) 
        (and 
            (and 
                (inside tinsel1 box1) 
                (under carpet1 tinsel1)
            ) 
            (and 
                (inside tinsel2 box1) 
                (under carpet1 tinsel2)
            )
        ) 
        (and 
            (ontop wreath1 carpet1) 
            (ontop stand1 carpet1) 
            (ontop topper1 carpet1) 
            (ontop skirt1 carpet1)
        ) 
        (inroom window1 living room) 
        (inroom carpet1 living room)
    )
    
    (:goal 
        (and 
            (and 
                (ontop ?stand1 ?carpet) 
                (ontop ?skirt1 ?stand1) 
                (under ?skirt1 ?tree1) 
                (inside ?tree1 ?stand1) 
                (nextto ?tree1 ?window1)
            ) 
            (and 
                (forall 
                    (?light - light) 
                    (ontop ?light ?tree1)
                ) 
                (forall 
                    (?bell - bell) 
                    (ontop ?bell ?tree1)
                ) 
                (forall 
                    (?tinsel - tinsel) 
                    (ontop ?tinsel ?tree1)
                )
            ) 
            (and 
                (ontop ?wreath1 ?tree1) 
                (ontop ?topper1 ?tree1)
            )
        )
    )
)