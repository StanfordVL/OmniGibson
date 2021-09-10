(define (problem making_tea_0)
    (:domain igibson)

    (:objects 
        teapot.n.01_1 - teapot.n.01
        tea_bag.n.01_1 - tea_bag.n.01
        lemon.n.01_1 - lemon.n.01
        knife.n.01_1 - knife.n.01
        cabinet.n.01_1 - cabinet.n.01
        electric_refrigerator.n.01_1 - electric_refrigerator.n.01
        stove.n.01_1 - stove.n.01
        floor.n.01_1 - floor.n.01
        agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (inside teapot.n.01_1 cabinet.n.01_1) 
        (inside tea_bag.n.01_1 cabinet.n.01_1) 
        (inside lemon.n.01_1 electric_refrigerator.n.01_1) 
        (inside knife.n.01_1 cabinet.n.01_1) 
        (inroom cabinet.n.01_1 kitchen) 
        (inroom electric_refrigerator.n.01_1 kitchen) 
        (inroom stove.n.01_1 kitchen) 
        (inroom floor.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (sliced ?lemon.n.01_1) 
            (ontop ?teapot.n.01_1 ?stove.n.01_1) 
            (inside ?tea_bag.n.01_1 ?teapot.n.01_1) 
            (soaked ?tea_bag.n.01_1) 
            (toggled_on ?stove.n.01_1)
        )
    )
)
