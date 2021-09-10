(define (problem cleaning_a_car_0)
    (:domain igibson)

    (:objects
     	car.n.01_1 - car.n.01
    	floor.n.01_1 - floor.n.01
    	rag.n.01_1 - rag.n.01
    	shelf.n.01_1 - shelf.n.01
    	soap.n.01_1 - soap.n.01
    	bucket.n.01_1 - bucket.n.01
    	sink.n.01_1 - sink.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor car.n.01_1 floor.n.01_1) 
        (ontop rag.n.01_1 shelf.n.01_1) 
        (not 
            (soaked rag.n.01_1)
        ) 
        (ontop soap.n.01_1 shelf.n.01_1) 
        (dusty car.n.01_1) 
        (onfloor bucket.n.01_1 floor.n.01_1) 
        (inroom floor.n.01_1 garage) 
        (inroom sink.n.01_1 bathroom) 
        (inroom shelf.n.01_1 garage) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (not 
                (dusty ?car.n.01_1)
            ) 
            (inside ?soap.n.01_1 ?bucket.n.01_1) 
            (inside ?rag.n.01_1 ?bucket.n.01_1)
        )
    )
)