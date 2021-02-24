(define (problem packing_lunches_1)
    (:domain igibson)

    (:objects
    	shelf.n.01_1 - shelf.n.01
    	bag.n.01_1 - bag.n.01
    	water.n.06_1 - water.n.06
    	countertop.n.01_1 - countertop.n.01
    	apple.n.01_1 - apple.n.01
    	electric_refrigerator.n.01_1 - electric_refrigerator.n.01
    	hamburger.n.01_1 - hamburger.n.01
    	chocolate.n.02_1 - chocolate.n.02
    	basket.n.01_1 - basket.n.01
    )
    
    (:init 
        (inside bag.n.01_1 shelf.n.01_1)
        (ontop water.n.06_1 countertop.n.01_1) 
        (inside apple.n.01_1 electric_refrigerator.n.01_1) 
        (inside hamburger.n.01_1 electric_refrigerator.n.01_1) 
        (inside chocolate.n.02_1 shelf.n.01_1) 
        (ontop basket.n.01_1 countertop.n.01_1) 
        (inroom countertop.n.01_1 kitchen) 
        (inroom electric_refrigerator.n.01_1 kitchen) 
        (inroom shelf.n.01_1 kitchen)
    )
    
    (:goal 
        (and 
            (forn 
                (1) 
                (?basket.n.01 - basket.n.01) 
                (and 
                    (fornpairs 
                        (1)
                        (?bag.n.01 - bag.n.01) 
                        (?hamburger.n.01 - hamburger.n.01) 
                        (and 
                            (inside ?hamburger.n.01 ?bag.n.01) 
                            (inside ?bag.n.01 ?basket.n.01) 
                            (not 
                                (open ?bag.n.01)
                            )
                        )
                    ) 
                )
            ) 
            (fornpairs 
                (1) 
                (?basket.n.01 - basket.n.01) 
                (?water.n.06 - water.n.06) 
                (inside ?water.n.06 ?basket.n.01)
            ) 
            (fornpairs 
                (1) 
                (?basket.n.01 - basket.n.01) 
                (?apple.n.01 - apple.n.01) 
                (inside ?apple.n.01 ?basket.n.01)
            ) 
            (fornpairs 
                (1) 
                (?basket.n.01 - basket.n.01) 
                (?chocolate.n.02 - chocolate.n.02) 
                (inside ?chocolate.n.02 ?basket.n.01)
            ) 
            (forall 
                (?basket.n.01 - basket.n.01) 
                (and 
                    (ontop ?basket.n.01 ?countertop.n.01_1) 
                    (not 
                        (open ?basket.n.01)
                    )
                )
            )
        )
    )
)
