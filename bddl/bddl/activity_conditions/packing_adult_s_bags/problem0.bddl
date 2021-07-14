(define (problem packing_adult_s_bags_0)
    (:domain igibson)

    (:objects
        backpack.n.01_1 - backpack.n.01
        floor.n.01_1 floor.n.01_2 - floor.n.01
        hanger.n.02_1 - hanger.n.02
        bed.n.01_1 - bed.n.01
        makeup.n.01_1 makeup.n.01_2 - makeup.n.01
        jewelry.n.01_1 jewelry.n.01_2 - jewelry.n.01
        toothbrush.n.01_1 - toothbrush.n.01
        mouse.n.04_1 - mouse.n.04
        door.n.01_1 - door.n.01
        agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor backpack.n.01_1 floor.n.01_1) 
        (ontop hanger.n.02_1 bed.n.01_1) 
        (ontop makeup.n.01_1 bed.n.01_1) 
        (ontop makeup.n.01_2 bed.n.01_1) 
        (ontop toothbrush.n.01_1 bed.n.01_1) 
        (onfloor jewelry.n.01_1 floor.n.01_1) 
        (onfloor jewelry.n.01_2 floor.n.01_1) 
        (ontop mouse.n.04_1 bed.n.01_1) 
        (open door.n.01_1) 
        (inroom floor.n.01_1 bedroom) 
        (inroom floor.n.01_2 corridor) 
        (inroom door.n.01_1 bedroom) 
        (inroom bed.n.01_1 bedroom) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forall 
                (?jewelry.n.01 - jewelry.n.01) 
                (inside ?jewelry.n.01 ?backpack.n.01_1)
            ) 
            (forall 
                (?makeup.n.01 - makeup.n.01) 
                (inside ?makeup.n.01 ?backpack.n.01_1)
            ) 
            (inside ?toothbrush.n.01_1 ?backpack.n.01_1) 
            (inside ?mouse.n.04_1 ?backpack.n.01_1) 
            (onfloor ?backpack.n.01_1 ?floor.n.01_2)
        )
    )
)