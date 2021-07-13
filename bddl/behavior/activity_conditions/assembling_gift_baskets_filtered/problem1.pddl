(define 
    (problem assembling_gift_baskets_1)
    (:domain igibson)

    (:objects
        basket.n.01_1 - basket.n.01
    	table.n.02_1 - table.n.02
        wine_bottle.n.01_1 - wine_bottle.n.01
    	candy_cane.n.01_1 - candy_cane.n.01
        shelf.n.01_1 - shelf.n.01
    )
    
    (:init 
        (ontop basket.n.01_1 table.n.02_1) 
        (ontop wine_bottle.n.01_1 shelf.n.01_1) 
        (inside candy_cane.n.01_1 shelf.n.01_1) 
        (inroom table.n.02_1 living_room) 
        (inroom shelf.n.01_1 living_room) 
    )
    
    (:goal 
        (and
            (forall
                (?basket.n.01 - basket.n.01)
                (and
                    (exists
                        (?candy_cane.n.01 - candy_cane.n.01)
                        (inside ?candy_cane.n.01 ?basket.n.01)
                    )
                    (exists
                        (?wine_bottle.n.01 - wine_bottle.n.01)
                        (inside ?wine_bottle.n.01 ?basket.n.01)
                    )
                )
            )
        )
    )
)
