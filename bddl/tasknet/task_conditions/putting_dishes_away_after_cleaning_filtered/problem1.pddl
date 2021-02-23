(define (problem putting_dishes_away_after_cleaning_1)
    (:domain igibson)

    (:objects
     	countertop.n.01_1 countertop.n.01_2 - countertop.n.01
    	sink.n.01_1 - sink.n.01
    	cabinet.n.01_1 - cabinet.n.01
    	bowl.n.01_1 bowl.n.01_2 bowl.n.01_3 bowl.n.01_4 - bowl.n.01
    	shelf.n.01_1 - shelf.n.01
    )
    
    (:init 
        (ontop bowl.n.01_1 countertop.n.01_1) 
        (ontop bowl.n.01_2 countertop.n.01_1) 
        (ontop bowl.n.01_3 countertop.n.01_1) 
        (ontop bowl.n.01_4 countertop.n.01_1) 
        (inroom cabinet.n.01_1 kitchen) 
        (inroom sink.n.01_1 kitchen) 
        (inroom countertop.n.01_1 kitchen) 
        (inroom countertop.n.01_2 kitchen) 
        (inroom shelf.n.01_1 kitchen)
    )
    
    (:goal 
        (and 
            (forall 
                (?bowl.n.01 - bowl.n.01) 
                (ontop ?bowl.n.01 ?shelf.n.01_1)
            ) 
        )
    )
)
