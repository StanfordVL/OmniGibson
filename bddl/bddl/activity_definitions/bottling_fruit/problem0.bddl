(define (problem bottling_fruit_0)
    (:domain igibson)

    (:objects
     	strawberry.n.01_1 - strawberry.n.01
    	electric_refrigerator.n.01_1 - electric_refrigerator.n.01
    	peach.n.03_1 - peach.n.03
    	countertop.n.01_1 - countertop.n.01
    	jar.n.01_1 jar.n.01_2 - jar.n.01
        carving_knife.n.01_1 - carving_knife.n.01
    	cabinet.n.01_1 - cabinet.n.01
    	floor.n.01_1 - floor.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (inside strawberry.n.01_1 electric_refrigerator.n.01_1) 
        (inside peach.n.03_1 electric_refrigerator.n.01_1) 
        (not 
            (sliced strawberry.n.01_1)
        ) 
        (not 
            (sliced peach.n.03_1)
        ) 
        (ontop jar.n.01_1 countertop.n.01_1) 
        (ontop jar.n.01_2 countertop.n.01_1) 
        (ontop carving_knife.n.01_1 countertop.n.01_1) 
        (inroom countertop.n.01_1 kitchen) 
        (inroom cabinet.n.01_1 kitchen) 
        (inroom electric_refrigerator.n.01_1 kitchen) 
        (inroom floor.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (exists 
                (?jar.n.01 - jar.n.01) 
                (and 
                    (inside ?strawberry.n.01_1 ?jar.n.01) 
                    (not 
                        (inside ?peach.n.03_1 ?jar.n.01)
                    )
                )
            ) 
            (exists 
                (?jar.n.01 - jar.n.01) 
                (and 
                    (inside ?peach.n.03_1 ?jar.n.01) 
                    (not 
                        (inside ?strawberry.n.01_1 ?jar.n.01)
                    )
                )
            ) 
            (forall 
                (?jar.n.01 - jar.n.01) 
                (not 
                    (open ?jar.n.01)
                )
            ) 
            (sliced strawberry.n.01_1) 
            (sliced peach.n.03_1)
        )
    )
)