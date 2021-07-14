(define (problem cleaning_bedroom_0)
    (:domain igibson)

    (:objects
        floor.n.01_1 - floor.n.01
        apparel.n.01_1 apparel.n.01_2 - apparel.n.01
        bed.n.01_1 - bed.n.01
        jewelry.n.01_1 - jewelry.n.01
        perfume.n.02_1 - perfume.n.02
        painting.n.01_1 - painting.n.01
        vacuum.n.04_1 - vacuum.n.04
        hand_towel.n.01_1 - hand_towel.n.01
        sheet.n.03_1 - sheet.n.03
        cabinet.n.01_1 cabinet.n.01_2 - cabinet.n.01
        agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (dusty cabinet.n.01_1) 
        (dusty cabinet.n.01_2) 
        (ontop apparel.n.01_1 bed.n.01_1) 
        (ontop apparel.n.01_2 bed.n.01_1) 
        (onfloor jewelry.n.01_1 floor.n.01_1) 
        (onfloor perfume.n.02_1 floor.n.01_1) 
        (ontop painting.n.01_1 bed.n.01_1) 
        (not 
            (dusty vacuum.n.04_1)
        ) 
        (onfloor vacuum.n.04_1 floor.n.01_1) 
        (onfloor hand_towel.n.01_1 floor.n.01_1)
        (onfloor sheet.n.03_1 floor.n.01_1) 
        (inroom floor.n.01_1 bedroom) 
        (inroom bed.n.01_1 bedroom) 
        (inroom cabinet.n.01_1 bedroom) 
        (inroom cabinet.n.01_2 bedroom) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forall 
                (?apparel.n.01 - apparel.n.01) 
                (exists 
                    (?cabinet.n.01 - cabinet.n.01) 
                    (inside ?apparel.n.01 ?cabinet.n.01)
                )
            ) 
            (and 
                (inside ?jewelry.n.01_1 ?cabinet.n.01_1) 
                (inside ?perfume.n.02_1 ?cabinet.n.01_1)
            ) 
            (ontop ?sheet.n.03_1 ?bed.n.01_1) 
            (forall 
                (?cabinet.n.01 - cabinet.n.01) 
                (not 
                    (dusty ?cabinet.n.01)
                )
            ) 
            (nextto ?vacuum.n.04_1 ?bed.n.01_1) 
            (ontop ?painting.n.01_1 ?sheet.n.03_1)
        )
    )
)