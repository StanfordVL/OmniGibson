(define (problem cleaning_garage_0)
    (:domain igibson)

    (:objects
     	box.n.01_1 - box.n.01
    	floor.n.01_1 floor.n.01_2 - floor.n.01
    	newspaper.n.03_1 newspaper.n.03_2 - newspaper.n.03
    	bottle.n.01_1 bottle.n.01_2 - bottle.n.01
    	cabinet.n.01_1 - cabinet.n.01
    	broom.n.01_1 - broom.n.01
    	rag.n.01_1 - rag.n.01
    	table.n.02_1 - table.n.02
    	bin.n.01_1 - bin.n.01
    	sink.n.01_1 - sink.n.01
    	shelf.n.01_1 - shelf.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor box.n.01_1 floor.n.01_1) 
        (onfloor newspaper.n.03_1 floor.n.01_1) 
        (onfloor newspaper.n.03_2 floor.n.01_1) 
        (onfloor bottle.n.01_1 floor.n.01_1) 
        (onfloor bottle.n.01_2 floor.n.01_1) 
        (dusty floor.n.01_1) 
        (stained floor.n.01_1) 
        (dusty cabinet.n.01_1) 
        (onfloor broom.n.01_1 floor.n.01_1) 
        (ontop rag.n.01_1 table.n.02_1) 
        (onfloor bin.n.01_1 floor.n.01_1) 
        (inroom floor.n.01_1 garage) 
        (inroom cabinet.n.01_1 garage) 
        (inroom sink.n.01_1 storage_room) 
        (inroom shelf.n.01_1 storage_room) 
        (inroom table.n.02_1 storage_room) 
        (inroom floor.n.01_2 storage_room) 
        (onfloor agent.n.01_1 floor.n.01_2)
    )
    
    (:goal 
        (and 
            (not 
                (dusty ?floor.n.01_1)
            ) 
            (not 
                (dusty ?cabinet.n.01_1)
            ) 
            (not 
                (stained ?cabinet.n.01_1)
            ) 
            (forall 
                (?newspaper.n.03 - newspaper.n.03) 
                (or 
                    (inside ?newspaper.n.03 ?bin.n.01_1) 
                    (not 
                        (onfloor ?newspaper.n.03 ?floor.n.01_1)
                    )
                )
            ) 
            (forall 
                (?bottle.n.01 - bottle.n.01) 
                (ontop ?bottle.n.01 ?table.n.02_1)
            )
        )
    )
)