(define (problem cleaning_a_car_0)
    (:domain igibson)

    (:objects
     	car.n.01_1 - car.n.01
    	floor.n.01_1 - floor.n.01
    	bucket.n.01_1 - bucket.n.01
    	water.n.06_1 - water.n.06
    	rag.n.01_1 - rag.n.01
    	shelf.n.01_1 - shelf.n.01
    	soap.n.01_1 - soap.n.01
    )
    
    (:init 
        (dusty car.n.01_1) 
        (onfloor car.n.01_1 floor.n.01_1) 
        (onfloor bucket.n.01_1 floor.n.01_1) 
        (onfloor water.n.06_1 floor.n.01_1) 
        (ontop rag.n.01_1 shelf.n.01_1) 
        (not 
            (soaked rag.n.01_1)
        ) 
        (ontop soap.n.01_1 shelf.n.01_1) 
        (inroom floor.n.01_1 garage) 
        (inroom shelf.n.01_1 garage)
    )
    
    (:goal 
        (and 
            (not 
                (dusty ?car.n.01_1)
            ) 
            (and 
                (and 
                    (inside ?rag.n.01_1 ?bucket.n.01_1) 
                    (soaked ?rag.n.01_1)
                ) 
                (inside ?soap.n.01_1 ?bucket.n.01_1)
            )
        )
    )
)