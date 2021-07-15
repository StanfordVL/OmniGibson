(define (problem cleaning_sneakers_0)
    (:domain igibson)

    (:objects
     	gym_shoe.n.01_1 gym_shoe.n.01_2 gym_shoe.n.01_3 gym_shoe.n.01_4 - gym_shoe.n.01
            countertop.n.01_1 - countertop.n.01
            soap.n.01_1 - soap.n.01
            cabinet.n.01_1 - cabinet.n.01
            towel.n.01_1 - towel.n.01
            brush.n.02_1 - brush.n.02
            sink.n.01_1 - sink.n.01
            floor.n.01_1 floor.n.01_2 - floor.n.01
            table.n.02_1 - table.n.02
            agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor gym_shoe.n.01_1 floor.n.01_2) 
        (stained gym_shoe.n.01_1) 
        (onfloor gym_shoe.n.01_2 floor.n.01_2) 
        (stained gym_shoe.n.01_2) 
        (onfloor gym_shoe.n.01_3 floor.n.01_2) 
        (dusty gym_shoe.n.01_3) 
        (onfloor gym_shoe.n.01_4 floor.n.01_2) 
        (dusty gym_shoe.n.01_4) 
        (inside soap.n.01_1 cabinet.n.01_1) 
        (ontop towel.n.01_1 countertop.n.01_1) 
        (not 
            (stained towel.n.01_1)
        ) 
        (ontop brush.n.02_1 countertop.n.01_1) 
        (not 
            (stained brush.n.02_1)
        ) 
        (inroom sink.n.01_1 kitchen) 
        (inroom countertop.n.01_1 kitchen) 
        (inroom cabinet.n.01_1 kitchen) 
        (inroom floor.n.01_1 living_room) 
        (inroom table.n.02_1 living_room) 
        (inroom floor.n.01_2 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_2)
    )
    
    (:goal 
        (and 
            (ontop ?towel.n.01_1 ?countertop.n.01_1) 
            (nextto ?brush.n.02_1 ?towel.n.01_1) 
            (inside ?soap.n.01_1 ?sink.n.01_1) 
            (forall 
                (?gym_shoe.n.01 - gym_shoe.n.01) 
                (not 
                    (dusty ?gym_shoe.n.01)
                )
            ) 
            (forall 
                (?gym_shoe.n.01 - gym_shoe.n.01) 
                (not 
                    (stained ?gym_shoe.n.01)
                )
            ) 
            (forn 
                (2) 
                (?gym_shoe.n.01 - gym_shoe.n.01) 
                (nextto ?gym_shoe.n.01 ?table.n.02_1)
            ) 
            (forn 
                (2) 
                (?gym_shoe.n.01 - gym_shoe.n.01) 
                (under ?gym_shoe.n.01 ?table.n.02_1)
            ) 
            (forall 
                (?gym_shoe.n.01 - gym_shoe.n.01) 
                (onfloor ?gym_shoe.n.01 ?floor.n.01_1)
            )
        )
    )
)