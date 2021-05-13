(define (problem putting_away_Christmas_decorations_0)
    (:domain igibson)

    (:objects
     	wrapping.n.01_1 wrapping.n.01_2 wrapping.n.01_3 - wrapping.n.01
    	floor.n.01_1 - floor.n.01
    	bow.n.08_1 bow.n.08_2 bow.n.08_3 bow.n.08_4 - bow.n.08
    	table.n.02_1 - table.n.02
    	wreath.n.01_1 wreath.n.01_2 - wreath.n.01
    	candle.n.01_1 - candle.n.01
    	container.n.01_1 - container.n.01
    	bag.n.01_1 bag.n.01_2 bag.n.01_3 bag.n.01_4 - bag.n.01
    	cabinet.n.01_1 - cabinet.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor wrapping.n.01_1 floor.n.01_1) 
        (onfloor wrapping.n.01_2 floor.n.01_1) 
        (onfloor wrapping.n.01_3 floor.n.01_1) 
        (ontop bow.n.08_1 table.n.02_1) 
        (ontop bow.n.08_2 table.n.02_1) 
        (ontop bow.n.08_3 table.n.02_1) 
        (ontop bow.n.08_4 table.n.02_1) 
        (onfloor wreath.n.01_1 floor.n.01_1) 
        (onfloor wreath.n.01_2 floor.n.01_1) 
        (ontop candle.n.01_1 table.n.02_1) 
        (onfloor container.n.01_1 floor.n.01_1) 
        (onfloor bag.n.01_1 floor.n.01_1) 
        (onfloor bag.n.01_2 floor.n.01_1) 
        (onfloor bag.n.01_3 floor.n.01_1) 
        (onfloor bag.n.01_4 floor.n.01_1) 
        (inroom floor.n.01_1 living_room) 
        (inroom cabinet.n.01_1 living_room) 
        (inroom table.n.02_1 living_room) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (exists 
                (?bag.n.01 - bag.n.01) 
                (and 
                    (forall 
                        (?wrapping.n.01 - wrapping.n.01) 
                        (inside ?wrapping.n.01 ?bag.n.01)
                    ) 
                    (forall 
                        (?bow.n.08 - bow.n.08) 
                        (inside ?bow.n.08 ?bag.n.01)
                    ) 
                    (forall 
                        (?wreath.n.01 - wreath.n.01) 
                        (inside ?wreath.n.01 ?bag.n.01)
                    ) 
                    (forall 
                        (?candle.n.01 - candle.n.01) 
                        (inside ?candle.n.01 ?bag.n.01)
                    )
                )
            ) 
            (forall 
                (?bag.n.01 - bag.n.01) 
                (onfloor ?bag.n.01 ?floor.n.01_1)
            )
        )
    )
)