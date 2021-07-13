(define (problem putting_up_Christmas_decorations_inside_0)
    (:domain igibson)

    (:objects 
        christmas_tree.n.05_1 - christmas_tree.n.05
        wreath.n.01_1 - wreath.n.01
        bow.n.08_1 bow.n.08_2 bow.n.08_3 - bow.n.08
        candle.n.01_1 candle.n.01_2 - candle.n.01
        wrapping.n.01_1 wrapping.n.01_2 wrapping.n.01_3 - wrapping.n.01
        carton.n.02_1 - carton.n.02
        floor.n.01_1 - floor.n.01
        table.n.02_1 - table.n.02
        sofa.n.01_1 - sofa.n.01
        agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor christmas_tree.n.05_1 floor.n.01_1)
        (onfloor carton.n.02_1 floor.n.01_1)
        (inside wreath.n.01_1 carton.n.02_1)
        (inside bow.n.08_1 carton.n.02_1)
        (inside bow.n.08_2 carton.n.02_1)
        (inside bow.n.08_3 carton.n.02_1)
        (inside candle.n.01_1 carton.n.02_1)
        (inside candle.n.01_2 carton.n.02_1)
        (onfloor wrapping.n.01_1 floor.n.01_1)
        (onfloor wrapping.n.01_2 floor.n.01_1)
        (onfloor wrapping.n.01_3 floor.n.01_1)
        (inroom floor.n.01_1 living_room)
        (inroom table.n.02_1 dining_room)
        (inroom sofa.n.01_1 living_room)
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forall 
                (?wrapping.n.01 - wrapping.n.01)
                (or 
                    (nextto ?wrapping.n.01 ?christmas_tree.n.05_1)
                    (under ?wrapping.n.01 ?christmas_tree.n.05_1)
                )
            )
            (forall 
                (?candle.n.01 - candle.n.01)
                (ontop ?candle.n.01 ?table.n.02_1)
            )
            (forn 
                (1)
                (?bow.n.08 - bow.n.08)
                (ontop ?bow.n.08 ?table.n.02_1)
            )
            (forn
                (2)
                (?bow.n.08 - bow.n.08)
                (ontop ?bow.n.08 ?sofa.n.01_1)
            )
            (ontop ?wreath.n.01_1 ?table.n.02_1)
        )
    )
)