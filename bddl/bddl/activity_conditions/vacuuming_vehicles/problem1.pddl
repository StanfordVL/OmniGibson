(define (problem vacuuming_vehicles_1)
    (:domain igibson)

    (:objects
     	motor_vehicle1 - motor_vehicle
    	floor1 - floor
    	car1 - car
    	mat1 mat2 mat3 mat4 - mat
    	vacuum1 - vacuum
    	cabinet1 - cabinet
    )
    
    (:init 
        (ontop motor_vehicle1 floor1) 
        (ontop car1 floor1) 
        (inside mat1 car1) 
        (inside mat2 car1) 
        (inside mat3 motor_vehicle1) 
        (inside mat4 motor_vehicle1) 
        (dusty mat1) 
        (dusty mat2) 
        (dusty mat3) 
        (dusty mat4) 
        (inside vacuum1 cabinet1) 
        (ontop mat1 floor1) 
        (ontop mat2 floor1) 
        (ontop mat3 floor1) 
        (ontop mat4 floor1) 
        (inroom floor1 garage) 
        (inroom cabinet1 garage)
    )
    
    (:goal 
        (and 
            (ontop ?motor_vehicle1 ?floor1) 
            (ontop ?car1 ?floor1) 
            (inside ?mat1 ?car1) 
            (inside ?mat2 ?car1) 
            (inside ?mat3 ?motor_vehicle1) 
            (inside ?mat4 ?motor_vehicle1) 
            (not 
                (dusty ?mat1)
            ) 
            (forall 
                (?mat - mat) 
                (not 
                    (dusty ?mat)
                )
            ) 
            (not 
                (inside ?vacuum1 ?cabinet1)
            )
        )
    )
)