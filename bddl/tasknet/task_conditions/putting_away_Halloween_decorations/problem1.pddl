(define (problem putting_away_Halloween_decorations_1)
    (:domain igibson)

    (:objects
     	chair1 chair2 - chair
    	carpet1 - carpet
    	table1 - table
    	skeleton1 skeleton2 - skeleton
    	hat1 hat2 - hat
    	pumpkin1 pumpkin2 - pumpkin
    	lamp1 lamp2 - lamp
    	plate1 - plate
    	candy1 candy2 candy3 candy4 candy5 candy6 candy7 - candy
    	shelf1 - shelf
    )
    
    (:init 
        (imply 
            (ontop chair1 carpet1) 
            (and 
                (nextto table1 chair1) 
                (nextto chair2 table1)
            )
        ) 
        (and 
            (ontop skeleton1 chair1) 
            (ontop skeleton2 chair2)
        ) 
        (and 
            (ontop hat1 skeleton1) 
            (under chair1 hat1)
        ) 
        (and 
            (ontop hat2 skeleton2) 
            (under chair2 hat2)
        ) 
        (and 
            (perished pumpkin1) 
            (ontop pumpkin1 carpet1) 
            (nextto pumpkin1 chair1) 
            (under pumpkin1 table1)
        ) 
        (and 
            (nextto pumpkin2 pumpkin1) 
            (perished pumpkin2) 
            (ontop pumpkin2 carpet1)
        ) 
        (and 
            (ontop lamp1 table1) 
            (ontop lamp2 table1)
        ) 
        (and 
            (ontop plate1 table1) 
            (inside candy1 plate1) 
            (inside candy2 plate1) 
            (inside candy3 plate1) 
            (inside candy4 plate1) 
            (inside candy5 plate1) 
            (inside candy6 plate1) 
            (inside candy7 plate1)
        ) 
        (and 
            (ontop candy1 table1) 
            (ontop candy2 table1) 
            (ontop candy3 table1) 
            (ontop candy4 table1) 
            (ontop candy5 table1) 
            (ontop candy6 table1) 
            (ontop candy7 table1)
        ) 
        (inroom carpet1 livingroom) 
        (inroom shelf1 livingroom) 
        (inroom chair1 livingroom) 
        (inroom chair2 livingroom) 
        (inroom table1 livingroom)
    )
    
    (:goal 
        (and 
            (exists 
                (?box - box) 
                (forpairs 
                    (?hat - hat) 
                    (?skeleton - skeleton) 
                    (and 
                        (inside ?skeleton ?box) 
                        (inside ?hat ?box)
                    )
                )
            ) 
            (exists 
                (?box - box) 
                (forall 
                    (?lamp - lamp) 
                    (inside ?lamp ?box)
                )
            ) 
            (exists 
                (?box - box) 
                (forall 
                    (?candy - candy) 
                    (inside ?candy ?box)
                )
            ) 
            (forall 
                (?pumpkin - pumpkin) 
                (inside ?pumpkin ?garbage1)
            ) 
            (forall 
                (?box - box) 
                (ontop ?box ?shelf)
            )
        )
    )
)