(define (problem collect_misplaced_items_0)
    (:domain igibson)

    (:objects
        gym_shoe.n.01_1 - gym_shoe.n.01
        necklace.n.01_1 - necklace.n.01
        notebook.n.01_1 - notebook.n.01
        sock.n.01_1 sock.n.01_2 - sock.n.01
        table.n.02_1 table.n.02_2 - table.n.02
        cabinet.n.01_1 - cabinet.n.01
        sofa.n.01_1 - sofa.n.01
        floor.n.01_1 floor.n.01_2 - floor.n.01
        agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (under gym_shoe.n.01_1 table.n.02_1) 
        (onfloor gym_shoe.n.01_1 floor.n.01_2) 
        (inside necklace.n.01_1 cabinet.n.01_1) 
        (under notebook.n.01_1 table.n.02_2) 
        (ontop sock.n.01_1 sofa.n.01_1) 
        (onfloor sock.n.01_2 floor.n.01_1) 
        (inroom table.n.02_1 living_room) 
        (inroom cabinet.n.01_1 living_room) 
        (inroom table.n.02_2 dining_room) 
        (inroom sofa.n.01_1 living_room) 
        (inroom floor.n.01_1 living_room) 
        (inroom floor.n.01_2 dining_room) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (ontop ?gym_shoe.n.01_1 ?table.n.02_2) 
            (ontop ?necklace.n.01_1 ?table.n.02_2) 
            (ontop ?notebook.n.01_1 ?table.n.02_2) 
            (forall 
                (?sock.n.01 - sock.n.01) 
                (ontop ?sock.n.01 ?table.n.02_2)
            )
        )
    )
)