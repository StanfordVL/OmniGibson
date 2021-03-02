(define (problem putting_away_Christmas_decorations_0)
    (:domain igibson)

    (:objects
        light_bulb.n.01_1 light_bulb.n.01_2 light_bulb.n.01_3 light_bulb.n.01_4 light_bulb.n.01_5 light_bulb.n.01_6 - light_bulb.n.01
        table.n.02_1 - table.n.02
        ball.n.01_1 ball.n.01_2 ball.n.01_3 - ball.n.01
        rug.n.01_1 - rug.n.01
        sofa.n.01_1 - sofa.n.01
        wreath.n.01_1 - wreath.n.01
        chair.n.01_1 - chair.n.01
        hook.n.05_1 - hook.n.05
        knickknack.n.01_1 knickknack.n.01_2 knickknack.n.01_3 - knickknack.n.01
        card.n.01_1 - card.n.01
        cord.n.01_1 - cord.n.01
    )

    (:init
        (and
            (ontop light_bulb.n.01_1 table.n.02_1)
            (ontop light_bulb.n.01_2 table.n.02_1)
            (ontop light_bulb.n.01_3 table.n.02_1)
        )
        (and
            (ontop ball.n.01_1 rug.n.01_1)
            (ontop ball.n.01_2 rug.n.01_1)
            (ontop ball.n.01_3 rug.n.01_1)
        )
        (and
            (ontop light_bulb.n.01_4 sofa.n.01_1)
            (ontop light_bulb.n.01_5 sofa.n.01_1)
            (ontop light_bulb.n.01_6 sofa.n.01_1)
        )
        (ontop wreath.n.01_1 chair.n.01_1)
        (nextto hook.n.05_1 wreath.n.01_1)
        (and
            (ontop knickknack.n.01_1 table.n.02_1)
            (under knickknack.n.01_2 table.n.02_1)
            (under knickknack.n.01_3 table.n.02_1)
        )
        (under card.n.01_1 sofa.n.01_1)
        (ontop cord.n.01_1 rug.n.01_1)
        (inroom chair.n.01_1 living_room)
        (inroom rug.n.01_1 living_room)
        (inroom table.n.02_1 living_room)
        (inroom sofa.n.01_1 living_room)
    )

    (:goal
        (and
            (exists
                (?container - container)
                (and
                    (forall
                        (?light_bulb.n.01 - light_bulb.n.01)
                        (inside ?light_bulb.n.01 ?container)
                    )
                    (forall
                        (?cord.n.01 - cord.n.01)
                        (inside ?cord.n.01 ?container)
                    )
                )
            )
            (exists
                (?container - container)
                (and
                    (forall
                        (?knickknack.n.01 - knickknack.n.01)
                        (inside ?knickknack.n.01 ?container)
                    )
                    (forall
                        (?ball.n.01 - ball.n.01)
                        (inside ?ball.n.01 ?container)
                    )
                )
            )
            (imply
                (exists
                    (?container - container)
                    (inside ?wreath.n.01_1 ?container)
                )
                (nextto ?hook.n.05_1 ?wreath.n.01_1)
            )
            (ontop ?card.n.01_1 ?table.n.02_1)
        )
    )
)
