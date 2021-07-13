(define (problem lifting_cauldon)
    (:domain igibson)

    (:objects 
        caldron.n.01_1 - caldron.n.01
        table.n.02_1 - table.n.02
        floor.n.01_1 - floor.n.01
        agent.n.01_1 - agent.n.01
    )

    (:init 
        (onfloor caldron.n.01_1 floor.n.01_1)
        (inroom table.n.02_1 living_room)
        (inroom floor.n.01_1 living_room)
        (onfloor agent.n.01_1 floor.n.01_1)
    )

    (:goal 
        (and 
            (ontop caldron.n.01_1 table.n.02_1)
        )
    )
)