(define (problem putting_dishes_away_after_cleaning_1)
    (:domain igibson)

    (:objects
     	countertop.n.01_1 - countertop.n.01
    	sink.n.01_1 - sink.n.01
    	cabinet.n.01_1 - cabinet.n.01
    	bowl.n.01_1 bowl.n.01_2 bowl.n.01_3 bowl.n.01_4 - bowl.n.01
    	plate.n.04_1 plate.n.04_2 plate.n.04_3 plate.n.04_4 - plate.n.04
    	tablefork.n.01_1 tablefork.n.01_2 tablefork.n.01_3 tablefork.n.01_4 - tablefork.n.01
    	table_knife.n.01_1 table_knife.n.01_2 table_knife.n.01_3 table_knife.n.01_4 - table_knife.n.01
    	shelf.n.01_1 - shelf.n.01
    )
    
    (:init 
        (scrubbed countertop.n.01_1) 
        (scrubbed sink.n.01_1) 
        (open cabinet.n.01_1) 
        (ontop bowl.n.01_1 countertop.n.01_1) 
        (ontop bowl.n.01_2 countertop.n.01_1) 
        (ontop bowl.n.01_3 countertop.n.01_1) 
        (ontop bowl.n.01_4 countertop.n.01_1) 
        (scrubbed bowl.n.01_1) 
        (scrubbed bowl.n.01_2) 
        (scrubbed bowl.n.01_3) 
        (scrubbed bowl.n.01_4) 
        (ontop plate.n.04_1 countertop.n.01_1) 
        (ontop plate.n.04_2 countertop.n.01_1) 
        (ontop plate.n.04_3 countertop.n.01_1) 
        (ontop plate.n.04_4 countertop.n.01_1) 
        (scrubbed plate.n.04_1) 
        (scrubbed plate.n.04_2) 
        (scrubbed plate.n.04_3) 
        (scrubbed plate.n.04_4) 
        (inside tablefork.n.01_1 sink.n.01_1) 
        (inside tablefork.n.01_2 sink.n.01_1) 
        (inside tablefork.n.01_3 sink.n.01_1) 
        (inside tablefork.n.01_4 sink.n.01_1) 
        (scrubbed tablefork.n.01_1) 
        (scrubbed tablefork.n.01_2) 
        (scrubbed tablefork.n.01_3) 
        (scrubbed tablefork.n.01_4) 
        (inside table_knife.n.01_1 sink.n.01_1) 
        (inside table_knife.n.01_2 sink.n.01_1) 
        (inside table_knife.n.01_3 sink.n.01_1) 
        (inside table_knife.n.01_4 sink.n.01_1) 
        (scrubbed table_knife.n.01_1) 
        (scrubbed table_knife.n.01_2) 
        (scrubbed table_knife.n.01_3) 
        (scrubbed table_knife.n.01_4) 
        (inroom cabinet.n.01_1 kitchen) 
        (inroom sink.n.01_1 kitchen) 
        (inroom countertop.n.01_1 kitchen) 
        (inroom shelf.n.01_1 kitchen)
    )
    
    (:goal 
        (and 
            (forall 
                (?bowl.n.01 - bowl.n.01) 
                (ontop ?bowl.n.01 ?shelf.n.01_1)
            ) 
            (forall 
                (?plate.n.04 - plate.n.04_) 
                (ontop ?plate.n.04 ?shelf.n.01_1)
            ) 
            (forall 
                (?tablefork.n.01 - tablefork.n.01) 
                (inside ?tablefork.n.01 ?cabinet.n.01_1)
            ) 
            (forall 
                (?table_knife.n.01 - table_knife.n.01) 
                (inside ?table_knife.n.01 ?cabinet.n.01_1)
            )
        )
    )
)
