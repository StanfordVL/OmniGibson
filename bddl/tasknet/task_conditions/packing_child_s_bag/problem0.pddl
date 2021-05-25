(define (problem packing_child_s_bag_0)
    (:domain igibson)

    (:objects
     	backpack.n.01_1 - backpack.n.01
        notebook.n.01_1 - notebook.n.01
        bracelet.n.02_1 - bracelet.n.02
        apple.n.01_1 - apple.n.01
    	floor.n.01_1 - floor.n.01
    	bed.n.01_1 - bed.n.01
        earphone.n.01_1 - earphone.n.01
    	sunglass.n.01_1 - sunglass.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor backpack.n.01_1 floor.n.01_1) 
        (ontop sunglass.n.01_1 bed.n.01_1) 
        (ontop bracelet.n.02_1 bed.n.01_1) 
        (onfloor notebook.n.01_1 floor.n.01_1) 
        (ontop apple.n.01_1 bed.n.01_1) 
        (ontop earphone.n.01_1 bed.n.01_1) 
        (inroom bed.n.01_1 bedroom) 
        (inroom floor.n.01_1 bedroom) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (nextto ?backpack.n.01_1 ?bed.n.01_1) 
            (inside ?sunglass.n.01_1 ?backpack.n.01_1) 
            (inside ?notebook.n.01_1 ?backpack.n.01_1) 
            (inside ?bracelet.n.02_1 ?backpack.n.01_1) 
            (inside ?apple.n.01_1 ?backpack.n.01_1) 
            (inside ?earphone.n.01_1 ?backpack.n.01_1)
        )
    )
)