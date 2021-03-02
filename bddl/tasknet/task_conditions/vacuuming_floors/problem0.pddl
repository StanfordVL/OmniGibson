(define (problem vacuuming_floors_0)
    (:domain igibson)

    (:objects
     	vacuum1 - vacuum
    	floor1 - floor
    	sofa1 - sofa
    	crumb1 crumb2 crumb3 crumb4 crumb5 - crumb
    	bed1 - bed
    	carpet1 - carpet
    	gym_shoe1 - gym_shoe
    )
    
    (:init 
        (and 
            (ontop vacuum1 floor1) 
            (ontop sofa1 floor1) 
            (ontop crumb1 floor1) 
            (ontop crumb2 floor1) 
            (ontop crumb3 floor1) 
            (dusty floor1)
        ) 
        (and 
            (ontop bed1 carpet1) 
            (ontop crumb4 carpet1) 
            (ontop crumb5 carpet1) 
            (ontop gym_shoe1 carpet1) 
            (dusty carpet1)
        ) 
        (inroom floor1 livingroom) 
        (inroom carpet1 bedroom) 
        (inroom bed1 bedroom) 
        (inroom sofa1 livingroom)
    )
    
    (:goal 
        (and 
            (forall 
                (?crumb - crumb) 
                (inside ?crumb ?vacuum1)
            ) 
            (nextto ?gym_shoe1 ?bed1) 
            (not 
                (dusty ?floor1)
            ) 
            (not 
                (dusty ?carpet1)
            )
        )
    )
)