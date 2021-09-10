(define (problem washing_pots_and_pans_0)
    (:domain igibson)

    (:objects
     	teapot.n.01_1 - teapot.n.01
        kettle.n.01_1 - kettle.n.01
    	pan.n.01_1 pan.n.01_2 pan.n.01_3 - pan.n.01
    	countertop.n.01_1 countertop.n.01_2 - countertop.n.01
    	sink.n.01_1 - sink.n.01
    	scrub_brush.n.01_1 - scrub_brush.n.01
    	soap.n.01_1 - soap.n.01
    	cabinet.n.01_1 cabinet.n.01_2 - cabinet.n.01
    	floor.n.01_1 - floor.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (ontop teapot.n.01_1 countertop.n.01_1) 
        (stained teapot.n.01_1) 
        (ontop kettle.n.01_1 countertop.n.01_2) 
        (stained kettle.n.01_1) 
        (ontop pan.n.01_1 countertop.n.01_1) 
        (stained pan.n.01_1) 
        (ontop pan.n.01_2 countertop.n.01_1) 
        (stained pan.n.01_2) 
        (ontop pan.n.01_3 countertop.n.01_2) 
        (stained pan.n.01_3) 
        (ontop scrub_brush.n.01_1 countertop.n.01_2) 
        (soaked scrub_brush.n.01_1) 
        (inside soap.n.01_1 sink.n.01_1) 
        (inroom cabinet.n.01_1 kitchen) 
        (inroom cabinet.n.01_2 kitchen) 
        (inroom sink.n.01_1 kitchen) 
        (inroom countertop.n.01_1 kitchen) 
        (inroom floor.n.01_1 kitchen) 
        (inroom countertop.n.01_2 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forall 
                (?pan.n.01 - pan.n.01) 
                (and 
                    (not 
                        (stained ?pan.n.01)
                    ) 
                    (exists 
                        (?cabinet.n.01 - cabinet.n.01) 
                        (inside ?pan.n.01 ?cabinet.n.01)
                    )
                )
            ) 
            (forall 
                (?kettle.n.01 - kettle.n.01) 
                (and 
                    (not 
                        (stained ?kettle.n.01)
                    ) 
                    (exists 
                        (?cabinet.n.01 - cabinet.n.01) 
                        (inside ?kettle.n.01 ?cabinet.n.01)
                    )
                )
            )
            (forall 
                (?teapot.n.01 - teapot.n.01) 
                (and 
                    (not 
                        (stained ?teapot.n.01)
                    ) 
                    (exists 
                        (?cabinet.n.01 - cabinet.n.01) 
                        (inside ?teapot.n.01 ?cabinet.n.01)
                    )
                )
            )
        )
    )
)