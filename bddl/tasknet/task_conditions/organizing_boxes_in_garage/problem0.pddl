(define (problem organizing_boxes_in_garage_0)
    (:domain igibson)

    (:objects
     	carton.n.02_1 carton.n.02_2 - carton.n.02
    	floor.n.01_1 - floor.n.01
    	ball.n.01_1 ball.n.01_2 - ball.n.01
    	plate.n.04_1 plate.n.04_2 plate.n.04_3 - plate.n.04
    	cabinet.n.01_1 - cabinet.n.01
    	saucepan.n.01_1 - saucepan.n.01
    	shelf.n.01_1 - shelf.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor carton.n.02_1 floor.n.01_1) 
        (onfloor carton.n.02_2 floor.n.01_1) 
        (onfloor ball.n.01_1 floor.n.01_1) 
        (onfloor ball.n.01_2 floor.n.01_1) 
        (inside plate.n.04_1 shelf.n.01_1) 
        (inside plate.n.04_2 shelf.n.01_1) 
        (inside plate.n.04_3 shelf.n.01_1) 
        (inside saucepan.n.01_1 shelf.n.01_1) 
        (inroom floor.n.01_1 garage) 
        (inroom shelf.n.01_1 garage) 
        (inroom cabinet.n.01_1 garage) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (exists 
                (?carton.n.02 - carton.n.02) 
                (and 
                    (forall 
                        (?ball.n.01 - ball.n.01) 
                        (inside ?ball.n.01 ?carton.n.02)
                    ) 
                    (forall 
                        (?plate.n.04 - plate.n.04) 
                        (inside ?plate.n.04 ?carton.n.02)
                    ) 
                    (inside ?saucepan.n.01_1 ?carton.n.02)
                )
            ) 
            (forall 
                (?carton.n.02 - carton.n.02) 
                (onfloor ?carton.n.02 floor.n.01_1)
            )
        )
    )
)