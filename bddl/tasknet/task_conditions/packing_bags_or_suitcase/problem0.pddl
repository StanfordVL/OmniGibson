(define (problem packing_bags_or_suitcase_0)
    (:domain igibson)

    (:objects
     	backpack.n.01_1 - backpack.n.01
    	floor.n.01_1 - floor.n.01
    	toothbrush.n.01_1 - toothbrush.n.01
    	bed.n.01_1 - bed.n.01
    	shampoo.n.01_1 - shampoo.n.01
    	hardback.n.01_1 - hardback.n.01
    	underwear.n.01_1 underwear.n.01_2 - underwear.n.01
    	toothpaste.n.01_1 - toothpaste.n.01
    	door.n.01_1 - door.n.01
    	window.n.01_1 - window.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor backpack.n.01_1 floor.n.01_1) 
        (ontop toothbrush.n.01_1 bed.n.01_1) 
        (ontop shampoo.n.01_1 bed.n.01_1) 
        (ontop hardback.n.01_1 bed.n.01_1) 
        (ontop underwear.n.01_1 bed.n.01_1) 
        (ontop underwear.n.01_2 bed.n.01_1) 
        (ontop toothpaste.n.01_1 bed.n.01_1) 
        (inroom bed.n.01_1 bedroom) 
        (inroom floor.n.01_1 bedroom) 
        (inroom door.n.01_1 bedroom) 
        (inroom window.n.01_1 bedroom) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forall 
                (?underwear.n.01 - underwear.n.01) 
                (inside ?underwear.n.01 ?backpack.n.01_1)
            ) 
            (inside ?toothbrush.n.01_1 ?backpack.n.01_1) 
            (inside ?shampoo.n.01_1 ?backpack.n.01_1) 
            (inside ?hardback.n.01_1 ?backpack.n.01_1) 
            (inside ?toothpaste.n.01_1 ?backpack.n.01_1) 
            (or 
                (ontop ?backpack.n.01_1 ?bed.n.01_1) 
                (onfloor ?backpack.n.01_1 ?floor.n.01_1)
            )
        )
    )
)