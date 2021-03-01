(define 
    (problem assembling_gift_baskets_1)
    (:domain igibson)

    (:objects
     	; basket1 basket2 basket3 basket4 - basket
    	table.n.02_1 - table.n.02
    	; notebook1 notebook2 notebook3 notebook4 - notebook
    	wine_bottle.n.01_1 wine_bottle.n.01_2 - wine_bottle.n.01
    	cheddar.n.02_1 cheddar.n.02_2 cheddar.n.02_3 cheddar.n.02_4 - cheddar.n.02
    	; photograph1 photograph2 - photograph
        envelope.n.01_1 envelope.n.01_2 - envelope.n.01
    	chocolate.n.02_1 chocolate.n.02_2 - chocolate.n.02
    	cracker.n.01_1 cracker.n.01_2 cracker.n.01_3 cracker.n.01_4 - cracker.n.01
    	candy_cane.n.01_1 candy_cane.n.01_2 - candy_cane.n.01
    	rug.n.01_1 - rug.n.01
    	sofa.n.01_1 - sofa.n.01
    )
    
    (:init 
    ; (and 
    ;     (ontop basket1 table.n.02_1) 
    ;     (ontop basket2 table.n.02_1) 
    ;     (ontop basket3 table.n.02_1) 
    ;     (ontop basket4 table.n.02_1)
    ; ) 
    ; (and 
    ;     (under notebook1 table.n.02_1) 
    ;     (under notebook2 table.n.02_1) 
    ;     (under notebook3 table.n.02_1) 
    ;     (under notebook4 table.n.02_1)
    ; ) 
        (ontop wine_bottle.n.01_1 table.n.02_1) 
        (ontop wine_bottle.n.01_2 table.n.02_1)
        (ontop cheddar.n.02_1 table.n.02_1) 
        (ontop cheddar.n.02_2 table.n.02_1) 
        (ontop cheddar.n.02_3 table.n.02_1) 
        (ontop cheddar.n.02_4 table.n.02_1)
    ; (and 
    ;     (ontop photograph1 table.n.02_1) 
    ;     (ontop photograph2 table.n.02_1)
    ; ) 
        (ontop envelope.n.01_1 table.n.02_1)
        (ontop envelope.n.01_2 table.n.02_1)
        (ontop chocolate.n.02_1 table.n.02_1) 
        (ontop chocolate.n.02_2 table.n.02_1)
        (ontop cracker.n.01_1 table.n.02_1) 
        (ontop cracker.n.01_2 table.n.02_1) 
        (ontop cracker.n.01_3 table.n.02_1) 
        (ontop cracker.n.01_4 table.n.02_1)
        (ontop candy_cane.n.01_1 table.n.02_1) 
        (ontop candy_cane.n.01_2 table.n.02_1)
        (inroom table.n.02_1 living_room) 
        (inroom rug.n.01_1 living_room) 
        (inroom sofa.n.01_1 living_room)
    )
    
    (:goal 
        (and 
            (forn 
                (2) 
                (?basket - basket) 
                (and 
                    (forn 
                        (2) 
                        (?notebook - notebook) 
                        (inside ?notebook ?basket)
                    ) 
                    (forn 
                        (2) 
                        (?cracker.n.01 - cracker.n.01) 
                        (inside ?cracker.n.01 ?basket)
                    ) 
                    (exists 
                        (?candy_cane.n.01 - candy_cane.n.01) 
                        (exists 
                            (?cracker.n.01 - cracker.n.01) 
                            (nextto ?candy_cane.n.01 ?cracker.n.01)
                        )
                    ) 
                    (exists 
                        (?cheddar.n.02 - cheddar.n.02) 
                        (exists 
                            (?candy_cane.n.01 - candy_cane.n.01) 
                            (nextto ?cheddar.n.02 ?candy_cane.n.01)
                        )
                    )
                )
            ) 
            (forn 
                (2) 
                (?basket - basket) 
                (and 
                    (forn 
                        (2) 
                        (?notebook - notebook) 
                        (inside ?notebook ?basket)
                    ) 
                    (exists 
                        (?wine_bottle.n.01 - wine_bottle.n.01) 
                        (exists 
                            (?cheddar.n.02 - cheddar.n.02) 
                            (nextto ?cheddar.n.02 ?wine_bottle.n.01)
                        )
                    ) 
                    (exists 
                        (?chocolate.n.02 - chocolate.n.02) 
                        (exists 
                            (?notebook - notebook) 
                            (ontop ?chocolate.n.02 ?notebook)
                        )
                    ) 
                    (exists 
                        (?cracker.n.01 - cracker.n.01) 
                        (exists 
                            (?chocolate.n.02 - chocolate.n.02) 
                            (nextto ?cracker.n.01 ?chocolate.n.02)
                        )
                    )
                )
            )
        )
    )
)
