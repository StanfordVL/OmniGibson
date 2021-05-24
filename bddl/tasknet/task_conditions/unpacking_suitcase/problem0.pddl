(define (problem unpacking_suitcase_0)
    (:domain igibson)

    (:objects
     	sock.n.01_1 sock.n.01_2 - sock.n.01
    	floor.n.01_1 - floor.n.01
        carton.n.02_1 - carton.n.02
        perfume.n.02_1 - perfume.n.02
        toothbrush.n.01_1 - toothbrush.n.01
        notebook.n.01_1 - notebook.n.01
    	sofa.n.01_1 - sofa.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor carton.n.02_1 floor.n.01_1) 
        (inside sock.n.01_1 carton.n.02_1) 
        (inside sock.n.01_2 carton.n.02_1) 
        (inside perfume.n.02_1 carton.n.02_1) 
        (inside toothbrush.n.01_1 carton.n.02_1) 
        (inside notebook.n.01_1 carton.n.02_1) 
        (inroom floor.n.01_1 living_room) 
        (inroom sofa.n.01_1 living_room) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (onfloor ?carton.n.02_1 ?floor.n.01_1) 
            (forall 
                (?sock.n.01 - sock.n.01) 
                (ontop ?sock.n.01 ?sofa.n.01_1)
            ) 
            (ontop ?perfume.n.02_1 ?sofa.n.01_1) 
            (ontop ?toothbrush.n.01_1 ?sofa.n.01_1) 
            (ontop ?notebook.n.01_1 ?sofa.n.01_1)
        )
    )
)