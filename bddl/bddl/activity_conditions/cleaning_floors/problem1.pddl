(define (problem cleaning_floors_1)
    (:domain igibson)

    (:objects
     	floor.n.01_1 - floor.n.01
    	broom.n.01_1 - broom.n.01
    	conditioner.n.03_1 - conditioner.n.03
    	shampoo.n.01_1 - shampoo.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (dusty floor.n.01_1) 
        (onfloor broom.n.01_1 floor.n.01_1) 
        (onfloor conditioner.n.03_1 floor.n.01_1) 
        (onfloor shampoo.n.01_1 floor.n.01_1) 
        (inroom floor.n.01_1 bathroom) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (not 
                (dusty ?floor.n.01_1)
            ) 
            (not 
                (onfloor ?conditioner.n.03_1 ?floor.n.01_1)
            ) 
            (not 
                (onfloor ?shampoo.n.01_1 ?floor.n.01_1)
            )
        )
    )
)