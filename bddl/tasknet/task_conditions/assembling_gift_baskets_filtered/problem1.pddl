(define 
    (problem assembling_gift_baskets_1)
    (:domain igibson)

    (:objects
        ; basket.n.01_1 basket.n.01_2 basket.n.01_3 basket.n.01_4 - basket.n.01
        basket.n.01_1 - basket.n.01
    	table.n.02_1 - table.n.02
    	wine_bottle.n.01_1 wine_bottle.n.01_2 wine_bottle.n.01_3 wine_bottle.n.01_4 - wine_bottle.n.01
    	cheddar.n.02_1 cheddar.n.02_2 cheddar.n.02_3 cheddar.n.02_4 - cheddar.n.02
    	; photograph1 photograph2 - photograph
        envelope.n.01_1 envelope.n.01_2 - envelope.n.01
    	chocolate.n.02_1 chocolate.n.02_2 - chocolate.n.02
    	cracker.n.01_1 cracker.n.01_2 cracker.n.01_3 cracker.n.01_4 - cracker.n.01
    	candy_cane.n.01_1 candy_cane.n.01_2 - candy_cane.n.01
    	table.n.02_1 - table.n.02
        shelf.n.01_1 shelf.n.01_2 - shelf.n.01
    )
    
    (:init 
        (ontop basket.n.01_1 table.n.02_1) 
        ; (ontop basket.n.01_2 table.n.02_1) 
        ; (ontop basket.n.01_3 table.n.02_1) 
        ; (ontop basket.n.01_4 table.n.02_1)
        ; (under notebook1 table.n.02_1) 
        ; (under notebook2 table.n.02_1) 
        ; (under notebook3 table.n.02_1) 
        ; (under notebook4 table.n.02_1)
        (ontop wine_bottle.n.01_1 shelf.n.01_1) 
        ; (ontop wine_bottle.n.01_2 shelf.n.01_1)
        ; (ontop wine_bottle.n.01_3 shelf.n.01_1)
        ; (ontop wine_bottle.n.01_4 shelf.n.01_1)
        ; (inside cheddar.n.02_1 shelf.n.01_2) 
        ; (ontop cheddar.n.02_2 shelf.n.01_1) 
        ; (ontop cheddar.n.02_3 shelf.n.01_1) 
        ; (ontop cheddar.n.02_4 shelf.n.01_1)
        ; (ontop photograph1 table.n.02_1) 
        ; (ontop photograph2 table.n.02_1)
        (inside envelope.n.01_1 shelf.n.01_2)
        ; (ontop envelope.n.01_2 table.n.02_1)
        ; (inside chocolate.n.02_1 shelf.n.01_1) 
        ; (ontop chocolate.n.02_2 shelf.n.01_1)
        (ontop cracker.n.01_1 shelf.n.01_2) 
        ; (ontop cracker.n.01_2 table.n.02_1) 
        ; (ontop cracker.n.01_3 table.n.02_1) 
        ; (ontop cracker.n.01_4 table.n.02_1)
        (inside candy_cane.n.01_1 shelf.n.01_2) 
        ; (ontop candy_cane.n.01_2 table.n.02_1)
        (inroom table.n.02_1 living_room) 
        ; (inroom table.n.02_2 living_room) 
        (inroom shelf.n.01_1 living_room) 
        (inroom shelf.n.01_2 living_room) 
    )
    
    (:goal 
        (and
            (forall
                (?basket.n.01 - basket.n.01)
                (and
                    (exists
                        (?cracker.n.01 - cracker.n.01)
                        (inside ?cracker.n.01 ?basket.n.01)
                    )
                    (exists
                        (?candy_cane.n.01 - candy_cane.n.01)
                        (inside ?candy_cane.n.01 ?basket.n.01)
                    )
                    (exists
                        (?wine_bottle.n.01 - wine_bottle.n.01)
                        (inside ?wine_bottle.n.01 ?basket.n.01)
                    )
                    (exists
                        (?cheddar.n.02 - cheddar.n.02)
                        (inside ?cheddar.n.02 ?basket.n.01)
                    )
                    (exists
                        (?envelope.n.01 - envelope.n.01)
                        (inside ?envelope.n.01 ?basket.n.01)
                    )
                )
            )
        )

        (and 
            (forn 
                (2) 
                (?basket.n.01 - basket.n.01) 
                (and 
                    ; (forn 
                    ;     (2) 
                    ;     (?notebook - notebook) 
                    ;     (inside ?notebook ?basket.n.01)
                    ; ) 
                    (forn 
                        (2) 
                        (?cracker.n.01 - cracker.n.01) 
                        (inside ?cracker.n.01 ?basket.n.01)
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
