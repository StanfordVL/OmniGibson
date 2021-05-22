(define (problem packing_food_for_work_1)
    (:domain igibson)

    (:objects
     	bag.n.01_1 - bag.n.01
    	countertop.n.01_1 - countertop.n.01
    	peach.n.03_1 - peach.n.03
    	electric_refrigerator.n.01_1 - electric_refrigerator.n.01
    	sandwich.n.01_1 - sandwich.n.01
    	floor.n.01_1 - floor.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (ontop bag.n.01_1 countertop.n.01_1) 
        (inside peach.n.03_1 electric_refrigerator.n.01_1) 
        (inside sandwich.n.01_1 electric_refrigerator.n.01_1) 
        (inroom electric_refrigerator.n.01_1 kitchen) 
        (inroom countertop.n.01_1 kitchen) 
        (inroom floor.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (ontop ?bag.n.01_1 ?countertop.n.01_1) 
            (inside ?peach.n.03_1 ?bag.n.01_1) 
            (inside ?sandwich.n.01_1 ?bag.n.01_1)
        )
    )
)