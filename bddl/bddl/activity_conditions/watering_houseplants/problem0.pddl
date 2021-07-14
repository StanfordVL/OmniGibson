(define (problem watering_houseplants_0)
    (:domain igibson)

    (:objects
     	pot_plant.n.01_1 pot_plant.n.01_2 pot_plant.n.01_3 - pot_plant.n.01
    	floor.n.01_1 floor.n.01_2 - floor.n.01
    	sink.n.01_1 - sink.n.01
    	table.n.02_1 - table.n.02
    	countertop.n.01_1 - countertop.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor pot_plant.n.01_1 floor.n.01_1) 
        (onfloor pot_plant.n.01_2 floor.n.01_1) 
        (onfloor pot_plant.n.01_3 floor.n.01_2) 
        (not 
            (soaked pot_plant.n.01_1)
        ) 
        (not 
            (soaked pot_plant.n.01_2)
        ) 
        (not 
            (soaked pot_plant.n.01_3)
        ) 
        (inroom table.n.02_1 dining_room) 
        (inroom floor.n.01_1 dining_room) 
        (inroom floor.n.01_2 kitchen) 
        (inroom countertop.n.01_1 kitchen) 
        (inroom sink.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_2)
    )
    
    (:goal 
        (and 
            (forall 
                (?pot_plant.n.01 - pot_plant.n.01) 
                (soaked ?pot_plant.n.01)
            )
        )
    )
)