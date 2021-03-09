(define (problem putting_away_Christmas_decorations_0)
    (:domain igibson)

    (:objects
        light_bulb.n.01_1 light_bulb.n.01_2 light_bulb.n.01_3 light_bulb.n.01_4 light_bulb.n.01_5 light_bulb.n.01_6 - light_bulb.n.01
        table.n.02_1 table.n.02_2 - table.n.02
        ball.n.01_1 ball.n.01_2 ball.n.01_3 - ball.n.01
        rug.n.01_1 - rug.n.01
        sofa.n.01_1 - sofa.n.01
        wreath.n.01_1 - wreath.n.01
        chair.n.01_1 - chair.n.01
        candle.n.01_1 candle.n.01_2 candle.n.01_3 - candle.n.01
        cord.n.01_1 - cord.n.01
        countertop.n.01_1 - countertop.n.01
        cabinet.n.01_1 - cabinet.n.01
    )

    (:init
        (ontop light_bulb.n.01_1 countertop.n.01_1)
        (ontop light_bulb.n.01_2 countertop.n.01_1)
        (ontop light_bulb.n.01_3 countertop.n.01_1)
        (ontop wreath.n.01_1 table.n.02_2)
        (ontop ball.n.01_1 rug.n.01_1)
        (ontop ball.n.01_2 rug.n.01_1)
        (ontop ball.n.01_3 rug.n.01_1)
        (ontop candle.n.01_1 table.n.02_1)
        (under candle.n.01_2 table.n.02_1)
        (under candle.n.01_3 table.n.02_1)
        (ontop cord.n.01_1 rug.n.01_1)
        (inroom chair.n.01_1 living_room)
        (inroom rug.n.01_1 living_room)
        (inroom table.n.02_1 living_room)
        (inroom table.n.02_2 living_room)
        (inroom sofa.n.01_1 living_room)
        (inroom countertop.n.01_1 living_room)
        (inroom cabinet.n.01_1 living_room)
    )

    (:goal
        (and
            (forall
                (?light_bulb.n.01 - light_bulb.n.01)
                (inside ?light_bulb.n.01 ?cabinet.n.01_1)
            )
            (forall
                (?wreath.n.01 - wreath.n.01)
                (inside ?wreath.n.01 ?cabinet.n.01_1)
            )
            (forall
                (?ball.n.01 - ball.n.01)
                (inside ?ball.n.01 ?cabinet.n.01_1)
            )
            (forall
                (?candle.n.01 - candle.n.01)
                (inside ?candle.n.01 ?cabinet.n.01_1)
            )
            (forall
                (?cord.n.01 - cord.n.01)
                (inside ?cord.n.01 ?cabinet.n.01_1)
            )
        )
    )
)
