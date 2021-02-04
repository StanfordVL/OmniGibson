(define (problem assembling_gift_baskets_1
    (:domain igibson)

    (:objects
     	basket1 basket2 basket3 basket4 - basket
    	table1 - table
    	scrapbook1 scrapbook2 scrapbook3 scrapbook4 - scrapbook
    	wine1 wine2 - wine
    	cheese1 cheese2 cheese3 cheese4 - cheese
    	card1 card2 - card
    	chocolate1 chocolate2 - chocolate
    	cracker1 cracker2 cracker3 cracker4 - cracker
    	candy1 candy2 - candy
    	carpet1 - carpet
    	sofa1 - sofa
    )
    
    (:init 
        (and 
            (ontop basket1 table1) 
            (ontop basket2 table1) 
            (ontop basket3 table1) 
            (ontop basket4 table1)
        ) 
        (and 
            (under scrapbook1 table1) 
            (under scrapbook2 table1) 
            (under scrapbook3 table1) 
            (under scrapbook4 table1)
        ) 
        (and 
            (ontop wine1 table1) 
            (ontop wine2 table1)
        ) 
        (and 
            (ontop cheese1 table1) 
            (ontop cheese2 table1) 
            (ontop cheese3 table1) 
            (ontop cheese4 table1)
        ) 
        (and 
            (ontop card1 table1) 
            (ontop card2 table1)
        ) 
        (and 
            (ontop chocolate1 table1) 
            (ontop chocolate2 table1)
        ) 
        (and 
            (ontop cracker1 table1) 
            (ontop cracker2 table1) 
            (ontop cracker3 table1) 
            (ontop cracker4 table1)
        ) 
        (and 
            (ontop candy1 table1) 
            (ontop candy2 table1)
        ) 
        (inroom table1 living room) 
        (inroom carpet1 living room) 
        (inroom sofa1 living room)
    )
    
    (:goal 
        (and 
            (forn 
                (2) 
                (?basket - basket) 
                (and 
                    (forn 
                        (2) 
                        (?scrapbook - scrapbook) 
                        (inside ?scrapbook ?basket)
                    ) 
                    (forn 
                        (2) 
                        (?cracker - cracker) 
                        (inside ?cracker ?basket)
                    ) 
                    (exists 
                        (?candy - candy) 
                        (exists 
                            (?cracker - cracker) 
                            (nextto ?candy ?cracker)
                        )
                    ) 
                    (exists 
                        (?cheese - cheese) 
                        (exists 
                            (?candy - candy) 
                            (nextto ?cheese ?candy)
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
                        (?scrapbook - scrapbook) 
                        (inside ?scrapbook ?basket)
                    ) 
                    (exists 
                        (?wine - wine) 
                        (exists 
                            (?cheese - cheese) 
                            (nextto ?cheese ?wine)
                        )
                    ) 
                    (exists 
                        (?chocolate - chocolate) 
                        (exists 
                            (?scrapbook - scrapbook) 
                            (ontop ?chocolate ?scrapbook)
                        )
                    ) 
                    (exists 
                        (?cracker - cracker) 
                        (exists 
                            (?chocolate - chocolate) 
                            (nextto ?cracker ?chocolate)
                        )
                    )
                )
            )
        )
    )
)