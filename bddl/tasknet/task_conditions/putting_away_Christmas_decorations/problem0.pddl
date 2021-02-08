(define (problem putting_away_Christmas_decorations_0)
    (:domain igibson)

    (:objects
        light1 light2 light3 light4 light5 light6 - light
        table1 - table
        ball1 ball2 ball3 - ball
        carpet1 - carpet
        sofa1 - sofa
        wreath1 - wreath
        sofa_chair1 - sofa_chair
        hook1 - hook
        knickknack1 knickknack2 knickknack3 - knickknack
        card1 - card
        cord1 - cord
    )
    
    (:init 
        (and 
            (ontop light1 table1) 
            (ontop light2 table1) 
            (ontop light3 table1)
        ) 
        (and 
            (ontop ball1 carpet1) 
            (ontop ball2 carpet1) 
            (ontop ball3 carpet1)
        ) 
        (and 
            (ontop light4 sofa1) 
            (ontop light5 sofa1) 
            (ontop light6 sofa1)
        ) 
        (ontop wreath1 sofa_chair1) 
        (nextto hook1 wreath1) 
        (and 
            (ontop knickknack1 table1) 
            (under knickknack2 table1) 
            (under knickknack3 table1)
        ) 
        (under card1 sofa1) 
        (ontop cord1 carpet1)
    )
    
    (:goal 
        (and 
            (exists 
                (?container - container) 
                (and 
                    (forall 
                        (?light - light) 
                        (inside ?light ?container)
                    ) 
                    (forall 
                        (?cord - cord) 
                        (inside ?cord ?container)
                    )
                )
            ) 
            (exists 
                (?container - container) 
                (and 
                    (forall 
                        (?knickknack - knickknack) 
                        (inside ?knickknack ?container)
                    ) 
                    (forall 
                        (?ball - ball) 
                        (inside ?ball ?container)
                    )
                )
            ) 
            (imply 
                (exists 
                    (?container - container) 
                    (inside ?wreath1 ?container)
                ) 
                (nextto ?hook1 ?wreath1)
            ) 
            (ontop ?card1 ?table1)
        )
    )
)