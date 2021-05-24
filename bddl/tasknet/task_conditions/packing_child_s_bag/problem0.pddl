(define (problem packing_child_s_bag_0)
    (:domain igibson)

    (:objects
     	briefcase.n.01_1 - briefcase.n.01
        notebook.n.01_1 - notebook.n.01
        bracelet.n.02_1 - bracelet.n.02
    	floor.n.01_1 - floor.n.01
    	bed.n.01_1 - bed.n.01
    	sunglass.n.01_1 - sunglass.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor briefcase.n.01_1 floor.n.01_1) 
        (ontop sunglass.n.01_1 bed.n.01_1) 
        (ontop bracelet.n.02_1 bed.n.01_1) 
        (onfloor notebook.n.01_1) 
        (inroom bed.n.01_1 bedroom) 
        (inroom floor.n.01_1 bedroom) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (nextto ?briefcase.n.01_1 ?bed.n.01_1) 
            (inside ?sunglass.n.01_1 ?briefcase.n.01_1) 
            (inside ?notebook.n.01_1 ?briefcase.n.01_1) 
            (inside ?bracelet.n.02_1 ?briefcase.n.01_1)
        )
    )
)