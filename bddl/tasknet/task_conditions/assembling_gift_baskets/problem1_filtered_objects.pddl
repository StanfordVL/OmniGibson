(define (problem assembling_gift_baskets_1)
    (:domain igibson)

    (:objects
     	hamper1 hamper2 hamper3 hamper4 - hamper
    	table1 - table
    	notebook1 notebook2 notebook3 notebook4 - notebook
    	wine_bottle1 wine_bottle2 - wine_bottle
    	cheddar1 cheddar2 cheddar3 cheddar4 - cheddar
    	photograph1 photograph2 - photograph
    	chocolate_box1 chocolate_box2 - chocolate_box
    	cracker_box1 cracker_box2 cracker_box3 cracker_box4 - cracker_box
    	candy_cane1 candy_cane2 - candy_cane
    	carpet1 - carpet
    	sofa1 - sofa
    )
    
    (:init 
        (and 
            (ontop hamper1 table1) 
            (ontop hamper2 table1) 
            (ontop hamper3 table1) 
            (ontop hamper4 table1)
        ) 
        (and 
            (under notebook1 table1) 
            (under notebook2 table1) 
            (under notebook3 table1) 
            (under notebook4 table1)
        ) 
        (and 
            (ontop wine_bottle1 table1) 
            (ontop wine_bottle2 table1)
        ) 
        (and 
            (ontop cheddar1 table1) 
            (ontop cheddar2 table1) 
            (ontop cheddar3 table1) 
            (ontop cheddar4 table1)
        ) 
        (and 
            (ontop photograph1 table1) 
            (ontop photograph2 table1)
        ) 
        (and 
            (ontop chocolate_box1 table1) 
            (ontop chocolate_box2 table1)
        ) 
        (and 
            (ontop cracker_box1 table1) 
            (ontop cracker_box2 table1) 
            (ontop cracker_box3 table1) 
            (ontop cracker_box4 table1)
        ) 
        (and 
            (ontop candy_cane1 table1) 
            (ontop candy_cane2 table1)
        ) 
        (inroom table1 living room) 
        (inroom carpet1 living room) 
        (inroom sofa1 living room)
    )
    
    (:goal 
        (and 
            (forn 
                (2) 
                (?hamper - hamper) 
                (and 
                    (forn 
                        (2) 
                        (?notebook - notebook) 
                        (inside ?notebook ?hamper)
                    ) 
                    (forn 
                        (2) 
                        (?cracker_box - cracker_box) 
                        (inside ?cracker_box ?hamper)
                    ) 
                    (exists 
                        (?candy_cane - candy_cane) 
                        (exists 
                            (?cracker_box - cracker_box) 
                            (nextto ?candy_cane ?cracker_box)
                        )
                    ) 
                    (exists 
                        (?cheddar - cheddar) 
                        (exists 
                            (?candy_cane - candy_cane) 
                            (nextto ?cheddar ?candy_cane)
                        )
                    )
                )
            ) 
            (forn 
                (2) 
                (?hamper - hamper) 
                (and 
                    (forn 
                        (2) 
                        (?notebook - notebook) 
                        (inside ?notebook ?hamper)
                    ) 
                    (exists 
                        (?wine_bottle - wine_bottle) 
                        (exists 
                            (?cheddar - cheddar) 
                            (nextto ?cheddar ?wine_bottle)
                        )
                    ) 
                    (exists 
                        (?chocolate_box - chocolate_box) 
                        (exists 
                            (?notebook - notebook) 
                            (ontop ?chocolate_box ?notebook)
                        )
                    ) 
                    (exists 
                        (?cracker_box - cracker_box) 
                        (exists 
                            (?chocolate_box - chocolate_box) 
                            (nextto ?cracker_box ?chocolate_box)
                        )
                    )
                )
            )
        )
    )
)
