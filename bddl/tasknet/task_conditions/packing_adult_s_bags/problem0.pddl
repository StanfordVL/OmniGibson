(define (problem packing_adult_s_bags_0)
    (:domain igibson)

    (:objects
     	briefcase.n.01_1 - briefcase.n.01
    	floor.n.01_1 floor.n.01_2 - floor.n.01
    	hanger.n.02_1 - hanger.n.02
    	bed.n.01_1 - bed.n.01
    	earphone.n.01_1 - earphone.n.01
    	makeup.n.01_1 - makeup.n.01
    	toothbrush.n.01_1 - toothbrush.n.01
    	underwear.n.01_1 - underwear.n.01
    	book.n.02_1 - book.n.02
    	door.n.01_1 - door.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor briefcase.n.01_1 floor.n.01_1) 
        (ontop hanger.n.02_1 bed.n.01_1) 
        (ontop earphone.n.01_1 bed.n.01_1) 
        (ontop makeup.n.01_1 bed.n.01_1) 
        (ontop toothbrush.n.01_1 bed.n.01_1) 
        (onfloor underwear.n.01_1 floor.n.01_1) 
        (onfloor book.n.02_1 floor.n.01_1) 
        (open door.n.01_1) 
        (inroom floor.n.01_1 bedroom) 
        (inroom floor.n.01_2 corridor) 
        (inroom door.n.01_1 bedroom) 
        (inroom bed.n.01_1 bedroom) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (and 
                (inside ?underwear.n.01_1 ?briefcase.n.01_1) 
                (inside ?earphone.n.01_1 ?briefcase.n.01_1) 
                (inside ?makeup.n.01_1 ?briefcase.n.01_1) 
                (inside ?toothbrush.n.01_1 ?briefcase.n.01_1) 
                (inside ?book.n.02_1 ?briefcase.n.01_1)
            ) 
            (onfloor ?hanger.n.02_1 ?floor.n.01_1) 
            (onfloor ?briefcase.n.01_1 ?floor.n.01_2)
        )
    )
)