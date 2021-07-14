(define (problem cleaning_kitchen_cupboard_0)
    (:domain igibson)

    (:objects
     	cabinet.n.01_1 cabinet.n.01_2 - cabinet.n.01
        countertop.n.01_1 - countertop.n.01
    	piece_of_cloth.n.01_1 - piece_of_cloth.n.01
    	cleansing_agent.n.01_1 - cleansing_agent.n.01
    	bowl.n.01_1 bowl.n.01_2 - bowl.n.01
    	cup.n.01_1 cup.n.01_2 - cup.n.01
    	sink.n.01_1 - sink.n.01
    	floor.n.01_1 - floor.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (dusty cabinet.n.01_1) 
        (dusty cabinet.n.01_2) 
        (ontop piece_of_cloth.n.01_1 countertop.n.01_1) 
        (ontop cleansing_agent.n.01_1 countertop.n.01_1) 
        (inside bowl.n.01_2 cabinet.n.01_2) 
        (inside cup.n.01_1 cabinet.n.01_2) 
        (inside cup.n.01_2 cabinet.n.01_1) 
        (ontop bowl.n.01_1 countertop.n.01_1) 
        (inroom cabinet.n.01_1 kitchen) 
        (inroom cabinet.n.01_2 kitchen) 
        (inroom sink.n.01_1 kitchen) 
        (inroom floor.n.01_1 kitchen) 
        (inroom countertop.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forall 
                (?cabinet.n.01 - cabinet.n.01) 
                (not 
                    (dusty ?cabinet.n.01)
                )
            ) 
            (exists 
                (?cabinet.n.01 - cabinet.n.01) 
                (forall 
                    (?bowl.n.01 - bowl.n.01) 
                    (and 
                        (inside ?bowl.n.01 ?cabinet.n.01) 
                        (not 
                            (inside ?cup.n.01_1 ?cabinet.n.01)
                        )
                    )
                )
            ) 
            (exists 
                (?cabinet.n.01 - cabinet.n.01) 
                (forall 
                    (?cup.n.01 - cup.n.01) 
                    (and 
                        (inside ?cup.n.01 ?cabinet.n.01) 
                        (not 
                            (inside ?bowl.n.01_1 ?cabinet.n.01)
                        )
                    )
                )
            )
        )
    )
)