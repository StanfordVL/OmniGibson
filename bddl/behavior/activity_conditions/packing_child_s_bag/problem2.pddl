(define (problem packing_child_s_bag_2)
    (:domain igibson)

    (:objects
     	duffel_bag.n.01_1 - duffel_bag.n.01
    	bed.n.01_1 - bed.n.01
    	jersey.n.03_1 - jersey.n.03
    	floor.n.01_1 - floor.n.01
    	underwear.n.01_1 - underwear.n.01
    	jean.n.01_1 - jean.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (ontop duffel_bag.n.01_1 bed.n.01_1) 
        (onfloor jersey.n.03_1 floor.n.01_1) 
        (onfloor underwear.n.01_1 floor.n.01_1) 
        (onfloor jean.n.01_1 floor.n.01_1) 
        (inroom bed.n.01_1 bedroom) 
        (inroom floor.n.01_1 bedroom) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (ontop ?duffel_bag.n.01_1 ?bed.n.01_1) 
            (inside ?jean.n.01_1 ?duffel_bag.n.01_1) 
            (inside ?jersey.n.03_1 ?duffel_bag.n.01_1) 
            (inside ?underwear.n.01_1 ?duffel_bag.n.01_1)
        )
    )
)