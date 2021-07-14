(define (problem sweeping_garage_0)
    (:domain igibson)

    (:objects
        garden_tool1 garden_tool2 garden_tool3 - garden_tool
        floor1 - floor
        car1 - car
        bicycle1 bicycle2 - bicycle
        broom1 - broom
        wall1 - wall
        dustpan1 - dustpan
        shelf1 - shelf
    )
    
    (:init 
        (ontop garden_tool1 floor1) 
        (ontop garden_tool2 floor1) 
        (ontop garden_tool3 floor1) 
        (ontop car1 floor1) 
        (ontop bicycle1 floor1) 
        (ontop bicycle2 floor1) 
        (dusty floor1) 
        (nextto broom1 wall1) 
        (nextto dustpan1 wall1) 
        (inroom floor1 garage) 
        (inroom wall1 garage) 
        (inroom shelf1 garage)
    )
    
    (:goal 
        (and 
            (not 
                (dusty ?floor1)
            ) 
            (ontop ?car1 ?floor1) 
            (forall 
                (?garden_tool - garden_tool) 
                (ontop ?garden_tool ?shelf)
            ) 
            (forall 
                (?bicycle - bicycle) 
                (nextto ?bicycle ?wall1)
            ) 
            (and 
                (nextto ?broom1 ?wall1) 
                (nextto ?dustpan1 ?wall1)
            )
        )
    )
)