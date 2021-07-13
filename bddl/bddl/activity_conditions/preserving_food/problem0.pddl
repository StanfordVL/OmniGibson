(define (problem preserving_food_0)
    (:domain igibson)

    (:objects 
        strawberry.n.01_1 strawberry.n.01_2 - strawberry.n.01
        beef.n.02_1 - beef.n.02
        jar.n.01_1 - jar.n.01
        pan.n.01_1 - pan.n.01
        electric_refrigerator.n.01_1 - electric_refrigerator.n.01
        cabinet.n.01_1 - cabinet.n.01
        countertop.n.01_1 - countertop.n.01
        carving_knife.n.01_1 - carving_knife.n.01
        floor.n.01_1 - floor.n.01
        agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (ontop strawberry.n.01_1 countertop.n.01_1) 
        (ontop strawberry.n.01_2 countertop.n.01_1) 
        (ontop beef.n.02_1 countertop.n.01_1) 
        (ontop jar.n.01_1 countertop.n.01_1) 
        (open jar.n.01_1) 
        (ontop pan.n.01_1 countertop.n.01_1) 
        (ontop carving_knife.n.01_1 countertop.n.01_1) 
        (inroom electric_refrigerator.n.01_1 kitchen) 
        (inroom cabinet.n.01_1 kitchen) 
        (inroom countertop.n.01_1 kitchen) 
        (inroom floor.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forall 
                (?strawberry.n.01 - strawberry.n.01) 
                (sliced ?strawberry.n.01)
            ) 
            (forall 
                (?strawberry.n.01 - strawberry.n.01) 
                (cooked ?strawberry.n.01)
            ) 
            (forall 
                (?strawberry.n.01 - strawberry.n.01) 
                (inside ?strawberry.n.01 ?jar.n.01_1)
            ) 
            (not 
                (open ?jar.n.01_1)
            ) 
            (inside ?beef.n.02_1 ?electric_refrigerator.n.01_1) 
            (frozen ?beef.n.02_1)
        )
    )
)