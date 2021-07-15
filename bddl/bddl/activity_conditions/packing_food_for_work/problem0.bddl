(define (problem packing_food_for_work_0)
    (:domain igibson)

    (:objects
        carton.n.02_1 - carton.n.02
        countertop.n.01_1 - countertop.n.01
        sandwich.n.01_1 - sandwich.n.01
        electric_refrigerator.n.01_1 - electric_refrigerator.n.01
        apple.n.01_1 - apple.n.01
        snack_food.n.01_1 - snack_food.n.01
        cabinet.n.01_1 - cabinet.n.01
        juice.n.01_1 - juice.n.01
        floor.n.01_1 - floor.n.01
        door.n.01_1 - door.n.01
        agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor carton.n.02_1 floor.n.01_1) 
        (inside sandwich.n.01_1 electric_refrigerator.n.01_1) 
        (ontop apple.n.01_1 countertop.n.01_1) 
        (inside snack_food.n.01_1 cabinet.n.01_1) 
        (ontop juice.n.01_1 countertop.n.01_1) 
        (inroom countertop.n.01_1 kitchen) 
        (inroom electric_refrigerator.n.01_1 kitchen) 
        (inroom cabinet.n.01_1 kitchen) 
        (inroom floor.n.01_1 kitchen) 
        (inroom door.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (inside ?sandwich.n.01_1 ?carton.n.02_1) 
            (inside ?apple.n.01_1 ?carton.n.02_1) 
            (inside ?snack_food.n.01_1 ?carton.n.02_1) 
            (inside ?juice.n.01_1 ?carton.n.02_1) 
            (or 
                (onfloor ?carton.n.02_1 ?floor.n.01_1) 
                (ontop ?carton.n.02_1 ?countertop.n.01_1)
            )
        )
    )
)