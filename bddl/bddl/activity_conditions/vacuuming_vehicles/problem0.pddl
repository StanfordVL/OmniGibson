(define (problem vacuuming_vehicles_0)
    (:domain igibson)

    (:objects
     	upholstery1 upholstery2 - upholstery
    	car1 - car
    	mat1 mat2 - mat
    	floor1 - floor
    	fiber1 fiber2 fiber3 fiber4 - fiber
    	rock1 rock2 rock3 rock4 - rock
    	vacuum1 - vacuum
    	wheeled_vehicle1 - wheeled_vehicle
    )
    
    (:init 
        (inside upholstery1 car1) 
        (inside upholstery2 car1) 
        (inside mat1 car1) 
        (inside mat2 car1) 
        (ontop upholstery1 floor1) 
        (ontop upholstery2 floor1) 
        (ontop mat1 floor1) 
        (ontop mat2 floor1) 
        (dusty upholstery1) 
        (dusty upholstery2) 
        (dusty mat1) 
        (dusty mat2) 
        (ontop fiber4 floor1) 
        (ontop fiber3 floor1) 
        (ontop fiber2 floor1) 
        (ontop fiber1 floor1) 
        (ontop rock4 floor1) 
        (ontop rock3 floor1) 
        (ontop rock2 floor1) 
        (ontop rock1 floor1) 
        (ontop vacuum1 floor1) 
        (ontop wheeled_vehicle1 floor1) 
        (ontop car1 floor1) 
        (nextto car1 vacuum1) 
        (nextto wheeled_vehicle1 vacuum1) 
        (inside rock1 wheeled_vehicle1) 
        (inside rock2 wheeled_vehicle1) 
        (inside rock3 wheeled_vehicle1) 
        (inside rock4 wheeled_vehicle1) 
        (inside fiber1 car1) 
        (inside fiber2 car1) 
        (inside fiber3 car1) 
        (inside fiber4 car1) 
        (inroom floor1 garage)
    )
    
    (:goal 
        (and 
            (forall 
                (?mat - mat) 
                (scrubbed ?mat)
            ) 
            (forall 
                (?upholstery - upholstery) 
                (scrubbed ?upholstery)
            ) 
            (ontop ?vacuum1 ?floor1) 
            (ontop ?wheeled_vehicle1 ?floor1) 
            (ontop ?car1 ?floor1) 
            (forall 
                (?rock - rock) 
                (nextto ?rock ?wheeled_vehicle1)
            ) 
            (forall 
                (?fiber - fiber) 
                (inside ?fiber ?vacuum1)
            ) 
            (nextto ?car1 ?vacuum1) 
            (nextto ?wheeled_vehicle1 ?vacuum1)
        )
    )
)