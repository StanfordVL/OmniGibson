(define (problem packing_lunches_1)
    (:domain igibson)

    (:objects
    	shelf.n.01_1 - shelf.n.01
    	bag.n.01_1 bag.n.01_2 bag.n.01_3 bag.n.01_4 bag.n.01_5 bag.n.01_6 bag.n.01_7 bag.n.01_8 - bag.n.01
    	water.n.06_1 water.n.06_2 water.n.06_3 water.n.06_4 - water.n.06
    	countertop.n.01_1 - countertop.n.01
    	apple.n.01_1 apple.n.01_2 apple.n.01_3 apple.n.01_4 - apple.n.01
    	electrical_refrigerator.n.01_1 - electrical_refrigerator.n.01
    	hamburger.n.01_1 hamburger.n.01_2 hamburger.n.01_3 hamburger.n.01_4 - hamburger.n.01
    	plum.n.02_1 plum.n.02_2 plum.n.02_3 plum.n.02_4 - plum.n.02
    	dinner_napkin.n.01_1 dinner_napkin.n.01_2 dinner_napkin.n.01_3 dinner_napkin.n.01_4 - dinner_napkin.n.01
    	chocolate.n.02_1 chocolate.n.02_2 chocolate.n.02_3 chocolate.n.02_4 - chocolate.n.02
    	basket.n.01_1 basket.n.01_2 basket.n.01_3 basket.n.01_4 basket.n.01_5 basket.n.01_6 - basket.n.01
    )
    
    (:init 
        (ontop basket.n.01_5 shelf.n.01_1) 
        (ontop basket.n.01_6 shelf.n.01_1) 
        (inside bag.n.01_1 basket.n.01_5) 
        (under shelf.n.01_1 bag.n.01_1)
        (inside bag.n.01_2 basket.n.01_5) 
        (under shelf.n.01_1 bag.n.01_2)
        (inside bag.n.01_3 basket.n.01_5) 
        (under shelf.n.01_1 bag.n.01_3)
        (inside bag.n.01_4 basket.n.01_5) 
        (under shelf.n.01_1 bag.n.01_4)
        (inside bag.n.01_5 basket.n.01_5) 
        (under shelf.n.01_1 bag.n.01_5)
        (inside bag.n.01_6 basket.n.01_5) 
        (under shelf.n.01_1 bag.n.01_6)
        (inside bag.n.01_7 basket.n.01_5) 
        (under shelf.n.01_1 bag.n.01_7)
        (inside bag.n.01_8 basket.n.01_5) 
        (under shelf.n.01_1 bag.n.01_8)
        (ontop water.n.06_1 countertop.n.01_1) 
        (ontop water.n.06_2 countertop.n.01_1) 
        (ontop water.n.06_3 countertop.n.01_1) 
        (ontop water.n.06_4 countertop.n.01_1) 
        (inside apple.n.01_1 freezer1) 
        (inside apple.n.01_2 freezer1) 
        (inside apple.n.01_3 freezer1) 
        (inside apple.n.01_4 freezer1) 
        (inside hamburger.n.01_1 freezer1) 
        (inside hamburger.n.01_2 freezer1) 
        (inside hamburger.n.01_3 freezer1) 
        (inside hamburger.n.01_4 freezer1)
        (inside plum.n.02_1 basket.n.01_6) 
        (under shelf.n.01_1 plum.n.02_1)
        (inside plum.n.02_2 basket.n.01_6) 
        (under shelf.n.01_1 plum.n.02_2)
        (inside plum.n.02_3 basket.n.01_6) 
        (under shelf.n.01_1 plum.n.02_3)
        (inside plum.n.02_4 basket.n.01_6) 
        (under shelf.n.01_1 plum.n.02_4)
        (ontop dinner_napkin.n.01_1 shelf.n.01_1) 
        (ontop dinner_napkin.n.01_2 shelf.n.01_1) 
        (ontop dinner_napkin.n.01_3 shelf.n.01_1) 
        (ontop dinner_napkin.n.01_4 shelf.n.01_1) 
        (ontop chocolate.n.02_1 shelf.n.01_1) 
        (ontop chocolate.n.02_2 shelf.n.01_1) 
        (ontop chocolate.n.02_3 shelf.n.01_1) 
        (ontop chocolate.n.02_4 shelf.n.01_1)
        (ontop basket.n.01_1 countertop.n.01_1) 
        (ontop basket.n.01_2 countertop.n.01_1) 
        (ontop basket.n.01_3 countertop.n.01_1) 
        (ontop basket.n.01_4 countertop.n.01_1)
        (inroom countertop.n.01_1 kitchen) 
        (inroom electrical_refrigerator.n.01_1 kitchen) 
        (inroom shelf.n.01_1 kitchen)
    
    (:goal 
        (and 
            (forn 
                (4) 
                (?basket.n.01 - basket.n.01) 
                (and 
                    (fornpairs 
                        (4) 
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
                    (fornpairs 
                        (4) 
                        (?bag.n.01 - bag.n.01) 
                        (?plum.n.02 - plum.n.02) 
                        (and 
                            (inside ?plum.n.02 ?bag.n.01) 
                            (inside ?bag.n.01 ?basket.n.01) 
                            (not 
                                (open ?bag.n.01)
                            )
                        )
                    )
                )
            ) 
            (fornpairs 
                (4) 
                (?basket.n.01 - basket.n.01) 
                (?dinner_napkin.n.01 - dinner_napkin.n.01) 
                (inside ?dinner_napkin.n.01 ?basket.n.01)
            ) 
            (fornpairs 
                (4) 
                (?basket.n.01 - basket.n.01) 
                (?water.n.06 - water.n.06) 
                (inside ?water.n.06 ?basket.n.01)
            ) 
            (fornpairs 
                (4) 
                (?basket.n.01 - basket.n.01) 
                (?apple.n.01 - apple.n.01) 
                (inside ?apple.n.01 ?basket.n.01)
            ) 
            (fornpairs 
                (4) 
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
