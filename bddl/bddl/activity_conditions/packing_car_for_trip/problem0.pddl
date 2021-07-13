(define (problem packing_car_for_trip_0)
    (:domain igibson)

    (:objects 
        car.n.01_1 - car.n.01
        briefcase.n.01_1 - briefcase.n.01
        pencil_box.n.01_1 - pencil_box.n.01
        headset.n.01_1 - headset.n.01
        duffel_bag.n.01_1 - duffel_bag.n.01
        table.n.02_1 - table.n.02
        floor.n.01_1 floor.n.01_2 - floor.n.01
        agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor car.n.01_1 floor.n.01_1) 
        (onfloor briefcase.n.01_1 floor.n.01_2) 
        (ontop pencil_box.n.01_1 table.n.02_1) 
        (ontop headset.n.01_1 table.n.02_1) 
        (onfloor duffel_bag.n.01_1 floor.n.01_2) 
        (inroom floor.n.01_1 garage) 
        (inroom floor.n.01_2 storage_room) 
        (inroom table.n.02_1 storage_room) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (inside briefcase.n.01_1 car.n.01_1) 
            (inside pencil_box.n.01_1 car.n.01_1) 
            (inside headset.n.01_1 car.n.01_1) 
            (inside duffel_bag.n.01_1 car.n.01_1)
        )
    )
)