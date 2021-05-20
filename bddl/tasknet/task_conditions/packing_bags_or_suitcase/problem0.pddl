(define (problem packing_bags_or_suitcase_0)
    (:domain igibson)

    (:objects
     	briefcase.n.01_1 - briefcase.n.01
    	floor.n.01_1 - floor.n.01
    	toothbrush.n.01_1 - toothbrush.n.01
    	bed.n.01_1 - bed.n.01
    	sweater.n.01_1 - sweater.n.01
    	jean.n.01_1 - jean.n.01
    	underwear.n.01_1 underwear.n.01_2 - underwear.n.01
    	toothpaste.n.01_1 - toothpaste.n.01
    	door.n.01_1 - door.n.01
    	window.n.01_1 - window.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor briefcase.n.01_1 floor.n.01_1) 
        (ontop toothbrush.n.01_1 bed.n.01_1) 
        (ontop sweater.n.01_1 bed.n.01_1) 
        (ontop jean.n.01_1 bed.n.01_1) 
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
                (inside ?underwear.n.01 ?briefcase.n.01_1)
            ) 
            (inside ?toothbrush.n.01_1 ?briefcase.n.01_1) 
            (inside ?sweater.n.01_1 ?briefcase.n.01_1) 
            (inside ?jean.n.01_1 ?briefcase.n.01_1) 
            (inside ?toothpaste.n.01_1 ?briefcase.n.01_1) 
            (or 
                (ontop ?briefcase.n.01_1 ?bed.n.01_1) 
                (onfloor ?briefcase.n.01_1 ?floor.n.01_1)
            )
        )
    )
)