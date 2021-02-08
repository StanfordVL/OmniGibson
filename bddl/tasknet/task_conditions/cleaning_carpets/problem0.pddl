(define (problem cleaning_carpets_0)
    (:domain igibson)

    (:objects
     	carpet1 carpet2 - carpet
    	bed1 - bed
    	bedclothes1 - bedclothes
    	sock1 sock2 - sock
    	towel1 - towel
    	receptacle1 - receptacle
    	table1 - table
    	office_chair1 - office_chair
    	fodder1 fodder2 fodder3 fodder4 - fodder
    	fur1 - fur
    	coffee_table1 - coffee_table
    	sofa1 - sofa
    	writing_implement1 writing_implement2 - writing_implement
    	bookcase1 - bookcase
    	book1 - book
    	vacuum1 - vacuum
    )
    
    (:init 
        (and 
            (dusty carpet1) 
            (not 
                (scrubbed carpet1)
            )
        ) 
        (and 
            (ontop bed1 carpet1) 
            (ontop bedclothes1 carpet1) 
            (and 
                (ontop sock1 carpet1) 
                (ontop sock2 carpet1)
            ) 
            (ontop towel1 carpet1) 
            (nextto receptacle1 bed1) 
            (ontop table1 carpet2) 
            (nextto office_chair1 table1) 
            (ontop fodder1 carpet1) 
            (ontop fodder2 carpet1)
        ) 
        (and 
            (dusty carpet2) 
            (not 
                (scrubbed carpet2)
            )
        ) 
        (and 
            (ontop fur1 carpet2) 
            (ontop coffee_table1 carpet2) 
            (nextto sofa1 coffee_table1) 
            (nextto fur1 sofa1) 
            (and 
                (nextto writing_implement1 coffee_table1) 
                (nextto writing_implement2 coffee_table1)
            ) 
            (nextto bookcase1 sofa1) 
            (ontop book1 carpet2) 
            (ontop fodder3 carpet2) 
            (ontop fodder4 carpet2) 
            (ontop vacuum1 carpet2)
        ) 
        (inroom coffee_table1 livingroom) 
        (inroom sofa1 livingroom) 
        (inroom carpet1 bedroom) 
        (inroom carpet2 livingroom) 
        (inroom bed1 bedroom) 
        (inroom table1 bedroom) 
        (inroom office_chair1 bedroom)
    )
    
    (:goal 
        (and 
            (and 
                (scrubbed ?carpet1) 
                (not 
                    (dusty ?carpet1)
                ) 
                (scrubbed ?carpet2) 
                (not 
                    (dusty ?carpet2)
                )
            ) 
            (forall 
                (?fodder - fodder) 
                (inside ?fodder ?vacuum1)
            ) 
            (inside ?fur1 ?vacuum1) 
            (and 
                (forall 
                    (?sock - sock) 
                    (inside ?sock ?receptacle1)
                ) 
                (inside ?towel1 ?receptacle1) 
                (ontop ?bedclothes1 ?bed1) 
                (inside ?book1 ?bookcase1) 
                (forall 
                    (?writing_implement - writing_implement) 
                    (ontop ?writing_implement ?coffee_table1)
                )
            )
        )
    )
)