(define (problem washing_cars_or_other_vehicles_0)
    (:domain igibson)

    (:objects
     	soap.n.01_1 - soap.n.01
    	car.n.01_1 - car.n.01
    	floor.n.01_1 - floor.n.01
    	bucket.n.01_1 - bucket.n.01
    	rag.n.01_1 - rag.n.01
    	agent.n.01_1 - agent.n.01
        sink.n.01_1 - sink.n.01
    )
    
    (:init 
        (ontop soap.n.01_1 car.n.01_1) 
        (onfloor car.n.01_1 floor.n.01_1) 
        (ontop bucket.n.01_1 car.n.01_1) 
        (ontop rag.n.01_1 car.n.01_1) 
        (dusty car.n.01_1) 
        (stained car.n.01_1) 
        (inroom floor.n.01_1 garage) 
        (inroom sink.n.01_1 storage_room) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (not 
                (dusty car.n.01_1)
            ) 
            (not 
                (stained car.n.01_1)
            )
        )
    )
)