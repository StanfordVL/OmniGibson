(define (problem sweeping_floors_0)
    (:domain igibson)

    (:objects
     	broom1 - broom
    	floor1 floor2 - floor
    	dustpan1 - dustpan
    	lint1 lint2 lint3 lint4 lint5 lint6 lint7 - lint
    	crumb1 crumb2 crumb3 crumb4 crumb5 crumb6 crumb7 - crumb
    )
    
    (:init 
        (ontop broom1 floor1) 
        (ontop dustpan1 floor1) 
        (ontop lint1 floor1) 
        (ontop lint2 floor1) 
        (ontop lint3 floor1) 
        (ontop lint4 floor1) 
        (ontop lint5 floor1) 
        (ontop lint6 floor1) 
        (ontop lint7 floor1) 
        (ontop crumb1 floor2) 
        (ontop crumb2 floor2) 
        (ontop crumb3 floor2) 
        (ontop crumb4 floor2) 
        (ontop crumb5 floor2) 
        (ontop crumb6 floor2) 
        (ontop crumb7 floor2) 
        (nextto dustpan1 broom1) 
        (dusty ?floor1) 
        (dusty ?floor2) 
        (inroom floor1 bathroom) 
        (inroom floor2 kitchen)
    )
    
    (:goal 
        (and 
            (forall 
                (?lint - lint) 
                (inside ?lint ?dustpan1)
            ) 
            (forall 
                (?crumb - crumb) 
                (inside ?crumb ?dustpan1)
            ) 
            (forall 
                (?floor - floor) 
                (not 
                    (dusty ?floor)
                )
            )
        )
    )
)