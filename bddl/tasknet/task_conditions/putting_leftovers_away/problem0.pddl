(define (problem putting_leftovers_away_0)
    (:domain igibson)

    (:objects
     	pasta.n.02_1 pasta.n.02_2 pasta.n.02_3 pasta.n.02_4 - pasta.n.02
    	floor.n.01_1 - floor.n.01
    	sauce.n.01_1 sauce.n.01_2 sauce.n.01_3 sauce.n.01_4 - sauce.n.01
    	electric_refrigerator.n.01_1 - electric_refrigerator.n.01
    	countertop.n.01_1 - countertop.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (ontop pasta.n.02_1 countertop.n.01_1) 
        (ontop pasta.n.02_2 countertop.n.01_1) 
        (ontop pasta.n.02_3 countertop.n.01_1) 
        (ontop pasta.n.02_4 countertop.n.01_1) 
        (ontop sauce.n.01_1 countertop.n.01_1) 
        (ontop sauce.n.01_2 countertop.n.01_1) 
        (ontop sauce.n.01_3 countertop.n.01_1) 
        (ontop sauce.n.01_4 countertop.n.01_1) 
        (inroom countertop.n.01_1 kitchen) 
        (inroom electric_refrigerator.n.01_1 kitchen) 
        (inroom floor.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forall 
                (?pasta.n.02 - pasta.n.02) 
                (inside ?pasta.n.02 ?electric_refrigerator.n.01_1)
            ) 
            (forall 
                (?sauce.n.01 - sauce.n.01) 
                (inside ?sauce.n.01 ?electric_refrigerator.n.01_1)
            )
        )
    )
)