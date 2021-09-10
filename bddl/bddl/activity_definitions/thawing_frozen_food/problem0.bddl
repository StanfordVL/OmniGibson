(define (problem thawing_frozen_food_0)
    (:domain igibson)

    (:objects
     	date.n.08_1 - date.n.08
    	electric_refrigerator.n.01_1 - electric_refrigerator.n.01
    	olive.n.04_1 - olive.n.04
    	fish.n.02_1 fish.n.02_2 fish.n.02_3 fish.n.02_4 - fish.n.02
    	sink.n.01_1 - sink.n.01
    	floor.n.01_1 - floor.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (inside date.n.08_1 electric_refrigerator.n.01_1) 
        (inside olive.n.04_1 electric_refrigerator.n.01_1) 
        (inside fish.n.02_1 electric_refrigerator.n.01_1) 
        (inside fish.n.02_2 electric_refrigerator.n.01_1) 
        (inside fish.n.02_3 electric_refrigerator.n.01_1) 
        (inside fish.n.02_4 electric_refrigerator.n.01_1) 
        (frozen fish.n.02_1) 
        (frozen fish.n.02_2) 
        (frozen fish.n.02_3) 
        (frozen fish.n.02_4) 
        (frozen date.n.08_1) 
        (frozen olive.n.04_1) 
        (inroom sink.n.01_1 kitchen) 
        (inroom electric_refrigerator.n.01_1 kitchen) 
        (inroom floor.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (nextto ?date.n.08_1 ?fish.n.02_1) 
            (nextto ?fish.n.02_1 ?sink.n.01_1) 
            (nextto ?fish.n.02_2 ?sink.n.01_1) 
            (nextto ?fish.n.02_3 ?sink.n.01_1) 
            (nextto ?fish.n.02_4 ?sink.n.01_1) 
            (nextto ?olive.n.04_1 ?sink.n.01_1)
        )
    )
)